"""Elo Progress Tracker - Track best model Elo over time for each config.

This module provides evidence of steady improvement in model quality by recording
snapshots of the best model's Elo rating for each board configuration over time.

Usage:
    from app.coordination.elo_progress_tracker import (
        get_elo_progress_tracker,
        snapshot_all_configs,
        get_progress_report,
    )

    # Take a snapshot of all configs
    await snapshot_all_configs()

    # Get progress report
    report = get_progress_report("hex8_2p", days=7)
    print(f"Elo improvement: {report.elo_delta}")

December 31, 2025: Created to track training loop effectiveness.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Database path
from app.utils.paths import DATA_DIR
from app.coordination.event_utils import make_config_key

ELO_PROGRESS_DB = DATA_DIR / "elo_progress.db"

# All canonical configs
ALL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


@dataclass
class EloSnapshot:
    """A point-in-time snapshot of the best model's Elo for a config."""
    config_key: str
    timestamp: float
    best_model_id: str
    best_elo: float
    games_played: int
    vs_random_win_rate: float | None = None
    vs_heuristic_win_rate: float | None = None


@dataclass
class ProgressReport:
    """Progress report for a config over a time period."""
    config_key: str
    start_elo: float | None
    end_elo: float | None
    elo_delta: float | None
    start_time: datetime | None
    end_time: datetime | None
    num_snapshots: int
    snapshots: list[EloSnapshot]

    @property
    def is_improving(self) -> bool:
        """Check if Elo is trending upward."""
        if self.elo_delta is None:
            return False
        return self.elo_delta > 0

    @property
    def improvement_rate_per_day(self) -> float | None:
        """Elo improvement per day."""
        if self.elo_delta is None or self.start_time is None or self.end_time is None:
            return None
        days = (self.end_time - self.start_time).total_seconds() / 86400
        if days < 0.01:
            return None
        return self.elo_delta / days


class EloProgressTracker:
    """Tracks best model Elo progress over time for each config."""

    _instance: EloProgressTracker | None = None

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or ELO_PROGRESS_DB
        self._init_db()
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to EVALUATION_COMPLETED events to capture win rates."""
        try:
            from app.coordination.event_router import get_router
            from app.coordination.data_events import DataEventType

            router = get_router()
            router.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
            logger.info("[EloProgress] Subscribed to EVALUATION_COMPLETED events")
        except ImportError as e:
            logger.debug(f"[EloProgress] Event router not available: {e}")
        except Exception as e:
            logger.warning(f"[EloProgress] Failed to subscribe to events: {e}")

    def _on_evaluation_completed(self, event) -> None:
        """Record Elo snapshot immediately after evaluation with win rate breakdown.

        This method extracts win rates from EVALUATION_COMPLETED events and
        persists them to the elo_progress table for tracking improvement over time.

        January 2026: Added to fix training feedback loop - win rates were not
        being recorded because EloProgressTracker wasn't subscribed to events.
        """
        # Extract payload from RouterEvent objects
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            return

        # Get config key - event may use 'config_key' or 'config'
        config_key = payload.get("config_key") or payload.get("config")
        if not config_key:
            return

        # Extract optional model_id and Elo rating
        model_id = payload.get("model_id") or payload.get("model_path", "")
        elo = payload.get("elo", 0.0)
        games_played = payload.get("games_played", 0)

        # Extract win rates - these were previously never recorded
        vs_random_rate = payload.get("vs_random_rate")
        vs_heuristic_rate = payload.get("vs_heuristic_rate")

        # Only record if we have meaningful data
        if not model_id and not elo:
            logger.debug(f"[EloProgress] Skipping event with no model_id or elo: {config_key}")
            return

        # January 2026: Validate model_id is appropriate for config_key
        if not self._validate_model_for_config(model_id, config_key):
            logger.warning(
                f"[EloProgress] Skipping invalid model-config: model_id='{model_id}' "
                f"does not match config_key='{config_key}'"
            )
            return

        # March 2026: Normalize model_id to canonical NN identity so that
        # event-driven snapshots use the same model_id as periodic snapshots
        # from snapshot_all_configs(). Without this, events record IDs like
        # "canonical_hex8_4p:gumbel_mcts:d4" while periodic snapshots record
        # "ringrift_best_hex8_4p", causing inconsistent best_model_id in
        # elo_progress.db and confusing the progress report.
        canonical_id = _extract_canonical_nn_id(str(model_id))
        normalized_model_id = canonical_id if canonical_id else str(model_id)

        # Record the snapshot with win rate breakdown
        self.record_snapshot(
            config_key=config_key,
            best_model_id=normalized_model_id,
            best_elo=float(elo) if elo else 0.0,
            games_played=int(games_played) if games_played else 0,
            vs_random_win_rate=float(vs_random_rate) if vs_random_rate is not None else None,
            vs_heuristic_win_rate=float(vs_heuristic_rate) if vs_heuristic_rate is not None else None,
        )

        logger.debug(
            f"[EloProgress] Recorded from event: {config_key} @ {elo:.1f} Elo "
            f"(model_id={model_id} -> {normalized_model_id}, "
            f"vs_random={vs_random_rate}, vs_heuristic={vs_heuristic_rate})"
        )

    def _validate_model_for_config(self, model_id: str, config_key: str) -> bool:
        """Validate model_id is appropriate for config_key.

        January 2026: Added to prevent model-config mismatches that corrupt Elo tracking.
        This catches the bug where config_key was mistakenly passed as model_id.

        Rules:
        1. If model_id looks like a config key (e.g., "hex8_2p"), log ERROR and reject
        2. If model_id is a model path, extract config and verify it matches
        3. Baseline names ("heuristic", "random") are allowed through

        Args:
            model_id: The model identifier from the event
            config_key: The expected configuration key

        Returns:
            True if valid, False if mismatch detected
        """
        model_id = str(model_id).strip()

        # Empty model_id - allow for backward compat but log debug
        if not model_id:
            logger.debug(f"[EloProgress] Empty model_id for {config_key}, allowing")
            return True

        # Detect config-as-model_id bug: if model_id matches config pattern
        if self._looks_like_config_key(model_id):
            logger.error(
                f"[EloProgress] BUG DETECTED: config_key '{model_id}' used as model_id! "
                f"Expected a model path, got config key. This corrupts Elo tracking."
            )
            return False

        # Allow baseline names through
        baseline_names = {"heuristic", "random", "mcts", "policy_only", "nnue"}
        if model_id.lower() in baseline_names:
            return True

        # Feb 2026: Reject composite participant IDs that contain harness types.
        # These look like "canonical_square8_2p:heuristic:d2:p1" and come from
        # evaluation events where participant_id is used instead of model_id.
        # Recording these corrupts Elo progress (heuristic baseline Elo != NN Elo).
        if ":" in model_id:
            parts = model_id.split(":")
            if len(parts) >= 2 and parts[1].lower() in baseline_names:
                logger.warning(
                    f"[EloProgress] Rejecting composite baseline ID: '{model_id}' "
                    f"(harness type '{parts[1]}' is a baseline, not an NN model)"
                )
                return False

        # Validate model path matches config
        if self._is_model_path(model_id):
            detected_config = self._extract_config_from_path(model_id)
            if detected_config and detected_config != config_key:
                logger.warning(
                    f"[EloProgress] Model-config mismatch: model '{model_id}' "
                    f"appears to be for '{detected_config}', not '{config_key}'"
                )
                return False

        return True

    def _looks_like_config_key(self, value: str) -> bool:
        """Check if a value looks like a config key rather than a model path.

        Config keys have format: {board_type}_{num_players}p
        Examples: hex8_2p, square8_4p, hexagonal_3p

        Args:
            value: String to check

        Returns:
            True if it looks like a config key
        """
        import re
        # Match patterns like hex8_2p, square8_4p, hexagonal_3p, square19_2p
        config_pattern = r"^(hex8|hexagonal|square8|square19)_[234]p$"
        return bool(re.match(config_pattern, value))

    def _is_model_path(self, value: str) -> bool:
        """Check if a value looks like a model file path.

        Args:
            value: String to check

        Returns:
            True if it looks like a model path
        """
        # Check for path separators or .pth extension
        return "/" in value or "\\" in value or value.endswith(".pth")

    def _extract_config_from_path(self, model_path: str) -> str | None:
        """Extract config key from a model file path.

        Handles patterns like:
        - models/canonical_hex8_2p.pth -> hex8_2p
        - /path/to/ringrift_best_square8_4p.pth -> square8_4p
        - models/hex8_2p/checkpoint_epoch10.pth -> hex8_2p

        Args:
            model_path: Path to model file

        Returns:
            Config key if found, None otherwise
        """
        import re
        from pathlib import Path as P

        # Get filename without extension
        filename = P(model_path).stem

        # Try to extract config from filename patterns
        # Pattern 1: canonical_{board}_{n}p or ringrift_best_{board}_{n}p
        match = re.search(r"(hex8|hexagonal|square8|square19)_([234])p", filename)
        if match:
            return f"{match.group(1)}_{match.group(2)}p"

        # Pattern 2: Check parent directory name
        parent = P(model_path).parent.name
        match = re.search(r"(hex8|hexagonal|square8|square19)_([234])p", parent)
        if match:
            return f"{match.group(1)}_{match.group(2)}p"

        return None

    @classmethod
    def get_instance(cls) -> EloProgressTracker:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    best_model_id TEXT NOT NULL,
                    best_elo REAL NOT NULL,
                    games_played INTEGER DEFAULT 0,
                    vs_random_win_rate REAL,
                    vs_heuristic_win_rate REAL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_progress_config
                ON elo_progress(config_key, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_progress_time
                ON elo_progress(timestamp)
            """)
            conn.commit()

    def record_snapshot(
        self,
        config_key: str,
        best_model_id: str,
        best_elo: float,
        games_played: int = 0,
        vs_random_win_rate: float | None = None,
        vs_heuristic_win_rate: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an Elo snapshot for a config."""
        import json

        timestamp = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO elo_progress
                (config_key, timestamp, best_model_id, best_elo, games_played,
                 vs_random_win_rate, vs_heuristic_win_rate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config_key,
                timestamp,
                best_model_id,
                best_elo,
                games_played,
                vs_random_win_rate,
                vs_heuristic_win_rate,
                json.dumps(metadata) if metadata else None,
            ))
            conn.commit()

        logger.info(
            f"[EloProgress] Recorded {config_key}: {best_model_id} @ {best_elo:.1f} Elo"
        )

    def get_snapshots(
        self,
        config_key: str,
        since_timestamp: float | None = None,
        limit: int = 100,
    ) -> list[EloSnapshot]:
        """Get Elo snapshots for a config."""
        with sqlite3.connect(self.db_path) as conn:
            if since_timestamp:
                cursor = conn.execute("""
                    SELECT config_key, timestamp, best_model_id, best_elo,
                           games_played, vs_random_win_rate, vs_heuristic_win_rate
                    FROM elo_progress
                    WHERE config_key = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (config_key, since_timestamp, limit))
            else:
                cursor = conn.execute("""
                    SELECT config_key, timestamp, best_model_id, best_elo,
                           games_played, vs_random_win_rate, vs_heuristic_win_rate
                    FROM elo_progress
                    WHERE config_key = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (config_key, limit))

            return [
                EloSnapshot(
                    config_key=row[0],
                    timestamp=row[1],
                    best_model_id=row[2],
                    best_elo=row[3],
                    games_played=row[4],
                    vs_random_win_rate=row[5],
                    vs_heuristic_win_rate=row[6],
                )
                for row in cursor.fetchall()
            ]

    def get_latest_snapshot(self, config_key: str) -> EloSnapshot | None:
        """Get the most recent snapshot for a config."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT config_key, timestamp, best_model_id, best_elo,
                       games_played, vs_random_win_rate, vs_heuristic_win_rate
                FROM elo_progress
                WHERE config_key = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (config_key,))
            row = cursor.fetchone()

            if not row:
                return None

            return EloSnapshot(
                config_key=row[0],
                timestamp=row[1],
                best_model_id=row[2],
                best_elo=row[3],
                games_played=row[4],
                vs_random_win_rate=row[5],
                vs_heuristic_win_rate=row[6],
            )

    def get_progress_report(
        self,
        config_key: str,
        days: float = 7.0,
    ) -> ProgressReport:
        """Get a progress report for a config over the specified time period."""
        since = time.time() - (days * 86400)
        snapshots = self.get_snapshots(config_key, since_timestamp=since, limit=10000)

        if not snapshots:
            return ProgressReport(
                config_key=config_key,
                start_elo=None,
                end_elo=None,
                elo_delta=None,
                start_time=None,
                end_time=None,
                num_snapshots=0,
                snapshots=[],
            )

        start = snapshots[0]
        end = snapshots[-1]

        return ProgressReport(
            config_key=config_key,
            start_elo=start.best_elo,
            end_elo=end.best_elo,
            elo_delta=end.best_elo - start.best_elo,
            start_time=datetime.fromtimestamp(start.timestamp, tz=timezone.utc),
            end_time=datetime.fromtimestamp(end.timestamp, tz=timezone.utc),
            num_snapshots=len(snapshots),
            snapshots=snapshots,
        )

    def get_all_progress_summary(self, days: float = 7.0) -> dict[str, ProgressReport]:
        """Get progress reports for all configs."""
        return {
            f"{bt}_{np}p": self.get_progress_report(f"{bt}_{np}p", days=days)
            for bt, np in ALL_CONFIGS
        }


def get_elo_progress_tracker() -> EloProgressTracker:
    """Get the singleton EloProgressTracker."""
    return EloProgressTracker.get_instance()


def _extract_canonical_nn_id(participant_id: str) -> str | None:
    """Extract the canonical NN identity from any participant ID format.

    Groups composite IDs by their nn_id component so that different harness
    evaluations of the same model (e.g., canonical_hex8_2p:gumbel_mcts:d2 and
    canonical_hex8_2p:policy_only:d2) are recognized as the same model.

    Also normalizes canonical_* to ringrift_best_* and strips version suffixes
    (e.g., _v2, _v5_heavy) for consistent grouping.

    Returns None for baselines (none:random, heuristic, etc).
    """
    from app.training.composite_participant import normalize_nn_id

    pid_lower = participant_id.lower()

    # Filter out baselines
    if any(x in pid_lower for x in ["random", "heuristic", "dummy", "baseline", "none:"]):
        return None

    # Composite ID: extract nn_id part
    if ":" in participant_id:
        nn_id = participant_id.split(":")[0]
    else:
        nn_id = participant_id

    # Normalize canonical_ → ringrift_best_ and strip version suffixes for grouping
    nn_id = normalize_nn_id(nn_id, strip_version=True) or nn_id

    return nn_id if nn_id and nn_id.lower() != "none" else None


async def snapshot_all_configs() -> dict[str, EloSnapshot | None]:
    """Take a snapshot of the best model Elo for all configs.

    This queries the unified_elo.db to find the best-performing model
    for each config and records it in the progress tracker.

    March 2026: Fixed false regressions caused by fragmented participant IDs.
    The same physical model file (e.g., canonical_hex8_4p.pth) can appear in
    the Elo leaderboard under multiple participant_ids:
      - ringrift_best_hex8_4p (bare legacy ID)
      - canonical_hex8_4p (bare legacy ID)
      - canonical_hex8_4p:gumbel_mcts:d4 (composite ID)
      - ringrift_best_hex8_4p:gumbel_mcts:d4:p1 (composite with player suffix)
    When Elo is recalculated from match history, each participant_id gets a
    different rating despite being the same model. The tracker would pick
    whichever ID happened to be highest-rated, and if a different ID was picked
    on the next snapshot, it could show a -400 Elo "regression" that wasn't real.

    Fix: Group all leaderboard entries by their canonical NN identity using
    _extract_canonical_nn_id(), then use the MAX rating across all participant
    IDs that map to the same physical model. This ensures the reported Elo
    reflects the model's true best performance regardless of which participant
    ID variant the Elo system happened to create.
    """
    from app.training.elo_service import get_elo_service

    tracker = get_elo_progress_tracker()
    elo_service = get_elo_service()
    results: dict[str, EloSnapshot | None] = {}

    for board_type, num_players in ALL_CONFIGS:
        config_key = make_config_key(board_type, num_players)

        try:
            # Get leaderboard for this config
            leaderboard = elo_service.get_leaderboard(
                board_type=board_type,
                num_players=num_players,
                limit=50,  # Increased to capture fragmented entries
                min_games=1,  # Must have played at least one game
            )

            # Filter to actual models (not random/heuristic baselines)
            model_entries = [
                entry for entry in leaderboard
                if not any(x in entry.participant_id.lower() for x in
                          ["random", "heuristic", "dummy", "baseline", "none:"])
            ]

            if not model_entries:
                logger.debug(f"[EloProgress] No models found for {config_key}")
                results[config_key] = None
                continue

            # March 2026: Group entries by canonical NN identity to find the
            # best rating across ALL participant_id variants of the same model.
            # This prevents false regressions when the same model has different
            # Elo under different participant_ids (e.g., ringrift_best_hex8_4p
            # at 2138 vs canonical_hex8_4p:gumbel_mcts:d4 at 1598).
            model_groups: dict[str, list] = {}
            for entry in model_entries:
                canonical_id = _extract_canonical_nn_id(entry.participant_id)
                if canonical_id is None:
                    # Could not extract canonical ID -- treat as its own group
                    canonical_id = entry.participant_id
                if canonical_id not in model_groups:
                    model_groups[canonical_id] = []
                model_groups[canonical_id].append(entry)

            # For each physical model, find the MAX rating and total games
            best_canonical_id = None
            best_max_rating = float("-inf")
            best_total_games = 0

            for canonical_id, entries in model_groups.items():
                max_entry = max(entries, key=lambda e: e.rating)
                total_games = sum(e.games_played for e in entries)

                if max_entry.rating > best_max_rating:
                    best_max_rating = max_entry.rating
                    best_total_games = total_games
                    best_canonical_id = canonical_id

            if best_canonical_id is None:
                results[config_key] = None
                continue

            logger.debug(
                f"[EloProgress] {config_key}: best model={best_canonical_id} "
                f"(max_elo={best_max_rating:.1f} across "
                f"{len(model_groups.get(best_canonical_id, []))} participant IDs, "
                f"total_games={best_total_games})"
            )

            # Record the snapshot using the canonical ID for consistency
            # and the MAX rating across all aliases
            tracker.record_snapshot(
                config_key=config_key,
                best_model_id=best_canonical_id,
                best_elo=best_max_rating,
                games_played=best_total_games,
            )

            results[config_key] = EloSnapshot(
                config_key=config_key,
                timestamp=time.time(),
                best_model_id=best_canonical_id,
                best_elo=best_max_rating,
                games_played=best_total_games,
            )

        except Exception as e:
            logger.warning(f"[EloProgress] Failed to snapshot {config_key}: {e}")
            results[config_key] = None

    return results


def get_progress_report(config_key: str, days: float = 7.0) -> ProgressReport:
    """Get progress report for a config."""
    return get_elo_progress_tracker().get_progress_report(config_key, days=days)


def print_progress_summary(days: float = 7.0) -> None:
    """Print a summary of Elo progress for all configs."""
    tracker = get_elo_progress_tracker()
    summary = tracker.get_all_progress_summary(days=days)

    print(f"\n{'='*70}")
    print(f"Elo Progress Report (last {days:.1f} days)")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Start Elo':>10} {'End Elo':>10} {'Delta':>8} {'Rate/day':>10} {'Snapshots':>10}")
    print(f"{'-'*70}")

    for config_key, report in sorted(summary.items()):
        if report.num_snapshots == 0:
            print(f"{config_key:<15} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>10} {0:>10}")
        else:
            start = f"{report.start_elo:.0f}" if report.start_elo else "N/A"
            end = f"{report.end_elo:.0f}" if report.end_elo else "N/A"
            delta = f"{report.elo_delta:+.0f}" if report.elo_delta else "N/A"
            rate = f"{report.improvement_rate_per_day:.1f}" if report.improvement_rate_per_day else "N/A"
            trend = "↑" if report.is_improving else "↓" if report.elo_delta and report.elo_delta < 0 else "→"
            print(f"{config_key:<15} {start:>10} {end:>10} {delta:>8} {rate:>10} {report.num_snapshots:>10} {trend}")

    print(f"{'='*70}\n")


# CLI entry point
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Elo Progress Tracker")
    parser.add_argument("--snapshot", action="store_true", help="Take a snapshot of all configs")
    parser.add_argument("--report", action="store_true", help="Print progress report")
    parser.add_argument("--days", type=float, default=7.0, help="Days to include in report")
    parser.add_argument("--config", type=str, help="Specific config to report on")
    args = parser.parse_args()

    if args.snapshot:
        print("Taking Elo snapshots for all configs...")
        results = asyncio.run(snapshot_all_configs())
        for config_key, snapshot in sorted(results.items()):
            if snapshot:
                print(f"  {config_key}: {snapshot.best_model_id} @ {snapshot.best_elo:.1f} Elo")
            else:
                print(f"  {config_key}: No data")

    if args.report or not args.snapshot:
        if args.config:
            report = get_progress_report(args.config, days=args.days)
            print(f"\nProgress for {args.config}:")
            print(f"  Start Elo: {report.start_elo}")
            print(f"  End Elo: {report.end_elo}")
            print(f"  Delta: {report.elo_delta}")
            print(f"  Snapshots: {report.num_snapshots}")
        else:
            print_progress_summary(days=args.days)
