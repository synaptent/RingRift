"""Data Quality Tools for Training Pipeline.

Provides comprehensive quality checking for both database and NPZ training data:

1. DatabaseQualityChecker - validate game database schema and content integrity
2. TrainingDataValidator - validate NPZ feature/label quality and distributions
3. DataQualityReport - structured quality assessment results

This module helps identify data corruption, schema mismatches, and training data
anomalies before they cause training failures or degraded model performance.

Quick Start - Programmatic Usage:
    # Check database quality
    from app.training.data_quality import DatabaseQualityChecker

    checker = DatabaseQualityChecker()
    score = checker.get_quality_score("data/games/selfplay.db")

    if score < 0.5:
        print("Warning: Database quality is poor")
        if checker.last_report:
            print(checker.last_report)

    # Validate NPZ training data
    from app.training.data_quality import TrainingDataValidator

    validator = TrainingDataValidator()

    # Basic validation
    if validator.validate_npz_file("data/training/batch_001.npz"):
        print("NPZ structure is valid")

    # Detailed analysis
    stats = validator.check_feature_distribution("data/training/batch_001.npz")
    print(f"Feature channels: {stats.get('num_channels', 0)}")

    outliers = validator.detect_outliers("data/training/batch_001.npz")
    print(f"Outliers found: {outliers.get('num_outliers', 0)}")

Quick Start - CLI Usage:
    # Check a specific database
    python -m app.training.data_quality --db data/games/selfplay.db

    # Validate NPZ training data with detailed statistics
    python -m app.training.data_quality --npz data/training/batch_001.npz --detailed

    # Scan all discovered databases
    python -m app.training.data_quality --all

Integration with Training Pipeline:
    # Pre-training validation
    from app.training.data_quality import TrainingDataValidator

    validator = TrainingDataValidator()

    for npz_file in training_files:
        if not validator.validate_npz_file(npz_file):
            logger.error(f"Skipping invalid file: {npz_file}")
            continue

        # Proceed with training...
        train_on_data(npz_file)

Author: RingRift AI Training Team
Last updated: December 2025
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.numpy_utils import safe_load_npz

logger = logging.getLogger(__name__)


# =============================================================================
# Data Checksums (December 2025)
# =============================================================================


def compute_array_checksum(arr: np.ndarray) -> str:
    """Compute SHA256 checksum for a numpy array.

    Uses tobytes() for raw data + shape/dtype info for full reproducibility.

    Args:
        arr: NumPy array to checksum

    Returns:
        Hex digest of SHA256 hash
    """
    hasher = hashlib.sha256()
    # Include array shape and dtype in hash to detect shape mismatches
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    # For object arrays (like policy indices), serialize differently
    if arr.dtype == np.object_:
        # Serialize each element
        for item in arr.flat:
            if isinstance(item, np.ndarray):
                hasher.update(item.tobytes())
            else:
                hasher.update(str(item).encode())
    else:
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def compute_npz_checksums(data: dict[str, np.ndarray]) -> dict[str, str]:
    """Compute checksums for all arrays in an NPZ dataset.

    Args:
        data: Dictionary of array name -> array

    Returns:
        Dictionary of array name -> checksum hex digest
    """
    checksums = {}
    for name, arr in data.items():
        if isinstance(arr, np.ndarray):
            checksums[name] = compute_array_checksum(arr)
    return checksums


def verify_npz_checksums(
    npz_path: str | Path,
    expected_checksums: dict[str, str] | None = None,
) -> tuple[bool, dict[str, str], list[str]]:
    """Verify checksums for arrays in an NPZ file.

    If expected_checksums is provided, compare against those.
    Otherwise, look for embedded checksums in the NPZ file itself.

    Args:
        npz_path: Path to NPZ file
        expected_checksums: Optional dict of array name -> expected checksum

    Returns:
        Tuple of (all_valid, computed_checksums, errors)
    """
    errors = []
    computed = {}

    try:
        with safe_load_npz(npz_path) as data:
            # Look for embedded checksums if not provided
            if expected_checksums is None:
                if "data_checksums" in data:
                    checksums_arr = data["data_checksums"]
                    # Parse JSON from array
                    if isinstance(checksums_arr, np.ndarray):
                        expected_checksums = json.loads(str(checksums_arr))
                    else:
                        expected_checksums = {}

            # Compute checksums for all arrays
            for name in data.files:
                if name == "data_checksums":
                    continue  # Skip the checksums array itself
                arr = data[name]
                if isinstance(arr, np.ndarray):
                    computed[name] = compute_array_checksum(arr)

            # Verify against expected
            if expected_checksums:
                for name, expected in expected_checksums.items():
                    if name not in computed:
                        errors.append(f"Missing array: {name}")
                    elif computed[name] != expected:
                        errors.append(
                            f"Checksum mismatch for {name}: "
                            f"expected {expected[:16]}..., got {computed[name][:16]}..."
                        )

    except Exception as e:
        errors.append(f"Failed to load NPZ: {e}")

    return len(errors) == 0, computed, errors


def embed_checksums_in_save_kwargs(save_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add checksums to save_kwargs before saving NPZ.

    This computes checksums for all array data and adds a 'data_checksums'
    field containing the JSON-serialized checksums.

    Args:
        save_kwargs: Dictionary of data to save to NPZ

    Returns:
        Updated save_kwargs with data_checksums added
    """
    checksums = compute_npz_checksums(save_kwargs)
    checksums_json = json.dumps(checksums, sort_keys=True)
    save_kwargs["data_checksums"] = np.asarray(checksums_json)
    return save_kwargs


# =============================================================================
# Data Quality Report
# =============================================================================


@dataclass
class DataQualityReport:
    """Comprehensive quality assessment for a dataset.

    Attributes:
        database_path: Path to the database file (None for NPZ)
        total_games: Total number of games in database
        valid_games: Number of valid/complete games
        quality_score: Overall quality score (0.0 - 1.0)
        issues: List of identified issues
        recommendations: List of improvement suggestions
        metadata: Additional diagnostic information
    """
    database_path: str | None = None
    total_games: int = 0
    valid_games: int = 0
    quality_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            "DATA QUALITY REPORT",
            "=" * 70,
        ]

        if self.database_path:
            lines.append(f"Database: {self.database_path}")

        lines.extend([
            f"Total Games: {self.total_games:,}",
            f"Valid Games: {self.valid_games:,}",
            f"Quality Score: {self.quality_score:.2%}",
        ])

        if self.issues:
            lines.append("\nISSUES FOUND:")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("\nNo issues found!")

        if self.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        if self.metadata:
            lines.append("\nMETADATA:")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"  {key}: {value}")

        lines.append("=" * 70)
        return "\n".join(lines)


class QualityIssueLevel(Enum):
    """Severity level for quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Database Quality Checker
# =============================================================================


class DatabaseQualityChecker:
    """Validates game database schema and content integrity.

    Performs comprehensive quality checks on SQLite game databases:
    - Schema consistency verification
    - Data integrity checks (NULL values, foreign keys, etc.)
    - Game quality metrics (length distribution, win rate balance)
    - Completeness checks (missing data, corrupted entries)

    Example:
        checker = DatabaseQualityChecker()

        # Check schema
        schema_ok = checker.check_schema_consistency("games.db")

        # Check data integrity
        integrity_ok = checker.check_data_integrity("games.db")

        # Analyze game quality
        quality_ok = checker.check_game_quality("games.db")

        # Get overall score
        score = checker.get_quality_score("games.db")
    """

    # Expected schema for games table
    EXPECTED_SCHEMA = {
        "game_id": "TEXT PRIMARY KEY",
        "board_type": "TEXT NOT NULL",
        "num_players": "INTEGER NOT NULL",
        "winner": "INTEGER",
        "total_moves": "INTEGER NOT NULL",
        "game_status": "TEXT NOT NULL",
    }

    # Acceptable game length ranges by board type (min, max)
    GAME_LENGTH_RANGES = {
        "square8": (10, 500),
        "square19": (20, 1000),
        "hexagonal": (15, 800),
        "hex8": (10, 500),
    }

    def __init__(self):
        self.last_report: DataQualityReport | None = None

    def check_schema_consistency(self, db_path: str | Path) -> bool:
        """Verify database schema matches expected structure.

        Args:
            db_path: Path to SQLite database

        Returns:
            True if schema is consistent, False otherwise
        """
        db_path = Path(db_path)

        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            return False

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()

            # Get actual schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='games'")
            result = cursor.fetchone()

            if not result:
                logger.error("Games table not found in database")
                conn.close()
                return False

            schema_sql = result[0]

            # Check for required columns (basic validation)
            missing_columns = []
            for col in self.EXPECTED_SCHEMA.keys():
                if col not in schema_sql.lower():
                    missing_columns.append(col)

            conn.close()

            if missing_columns:
                logger.warning(f"Missing expected columns: {missing_columns}")
                return False

            logger.info(f"Schema validation passed for {db_path}")
            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def check_data_integrity(self, db_path: str | Path) -> bool:
        """Check for data integrity issues.

        Validates:
        - No NULL winners in completed games
        - Valid move counts (> 0)
        - Valid player indices
        - Consistent game status

        Args:
            db_path: Path to SQLite database

        Returns:
            True if integrity checks pass, False otherwise
        """
        db_path = Path(db_path)

        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            return False

        issues = []

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()

            # Check for NULL winners in completed games
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE game_status = 'completed' AND winner IS NULL"
            )
            null_winners = cursor.fetchone()[0]
            if null_winners > 0:
                issues.append(f"{null_winners} completed games with NULL winner")

            # Check for invalid move counts
            cursor.execute("SELECT COUNT(*) FROM games WHERE total_moves <= 0")
            invalid_moves = cursor.fetchone()[0]
            if invalid_moves > 0:
                issues.append(f"{invalid_moves} games with invalid move count")

            # Check for invalid player numbers
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE num_players < 2 OR num_players > 4"
            )
            invalid_players = cursor.fetchone()[0]
            if invalid_players > 0:
                issues.append(f"{invalid_players} games with invalid player count")

            # Check for games with winner >= num_players
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL AND winner >= num_players"
            )
            invalid_winner_idx = cursor.fetchone()[0]
            if invalid_winner_idx > 0:
                issues.append(f"{invalid_winner_idx} games with invalid winner index")

            # Check for negative winner indices
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL AND winner < 0"
            )
            negative_winners = cursor.fetchone()[0]
            if negative_winners > 0:
                issues.append(f"{negative_winners} games with negative winner index")

            conn.close()

            if issues:
                for issue in issues:
                    logger.warning(f"Integrity issue: {issue}")
                return False

            logger.info(f"Data integrity checks passed for {db_path}")
            return True

        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False

    def check_games_with_moves(self, db_path: str | Path) -> tuple[bool, dict[str, Any]]:
        """Check if games have move data (either in game_moves table or games.moves column).

        This is CRITICAL for training data export - games without moves
        will produce zero training samples, causing silent export failures.

        Supports both database schemas:
        - New schema: moves in separate game_moves table
        - Old schema: moves as JSON in games.moves column

        Args:
            db_path: Path to SQLite database

        Returns:
            Tuple of (passes_check, stats_dict) where stats_dict contains:
            - total_games: Total games in database
            - games_with_moves: Games that have move data
            - games_without_moves: Games missing move data
            - coverage_percent: Percentage of games with moves
            - schema_type: 'game_moves_table' or 'moves_column'
            - issue: Description of issue if check fails (None if passes)
        """
        db_path = Path(db_path)
        stats: dict[str, Any] = {
            "total_games": 0,
            "games_with_moves": 0,
            "games_without_moves": 0,
            "coverage_percent": 0.0,
            "schema_type": None,
            "issue": None,
        }

        if not db_path.exists():
            stats["issue"] = f"Database not found: {db_path}"
            return False, stats

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()

            # Check if game_moves table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
            )
            has_game_moves_table = cursor.fetchone() is not None

            # Check if games table has moves column (old schema)
            cursor.execute("PRAGMA table_info(games)")
            columns = [row[1] for row in cursor.fetchall()]
            has_moves_column = "moves" in columns

            if not has_game_moves_table and not has_moves_column:
                stats["issue"] = "No game_moves table and no moves column in games table"
                conn.close()
                return False, stats

            # Count total games
            cursor.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]
            stats["total_games"] = total_games

            if total_games == 0:
                stats["issue"] = "Database contains no games"
                conn.close()
                return False, stats

            # Count games with moves based on schema type
            if has_game_moves_table:
                # New schema: check game_moves table
                stats["schema_type"] = "game_moves_table"
                cursor.execute("""
                    SELECT COUNT(DISTINCT g.game_id)
                    FROM games g
                    WHERE EXISTS (
                        SELECT 1 FROM game_moves m WHERE m.game_id = g.game_id
                    )
                """)
                games_with_moves = cursor.fetchone()[0]
            else:
                # Old schema: check moves column is non-empty
                stats["schema_type"] = "moves_column"
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM games
                    WHERE moves IS NOT NULL AND moves != '' AND moves != '[]'
                """)
                games_with_moves = cursor.fetchone()[0]
            stats["games_with_moves"] = games_with_moves
            stats["games_without_moves"] = total_games - games_with_moves
            stats["coverage_percent"] = 100.0 * games_with_moves / total_games

            conn.close()

            # Fail if coverage is below threshold (10% = critical failure)
            if stats["coverage_percent"] < 10.0:
                stats["issue"] = (
                    f"CRITICAL: Only {stats['coverage_percent']:.1f}% of games have move data "
                    f"({games_with_moves}/{total_games}). Training export will fail silently!"
                )
                logger.error(stats["issue"])
                return False, stats

            # Warn if coverage is below 90%
            if stats["coverage_percent"] < 90.0:
                stats["issue"] = (
                    f"WARNING: Only {stats['coverage_percent']:.1f}% of games have move data "
                    f"({games_with_moves}/{total_games}). Some games may have been imported "
                    "incorrectly (e.g., JSONL without move extraction)."
                )
                logger.warning(stats["issue"])
                return False, stats

            logger.info(
                f"Games with moves: {games_with_moves}/{total_games} "
                f"({stats['coverage_percent']:.1f}%)"
            )
            return True, stats

        except Exception as e:
            stats["issue"] = f"Failed to check games with moves: {e}"
            logger.error(stats["issue"])
            return False, stats

    def check_game_quality(self, db_path: str | Path) -> bool:
        """Analyze game quality metrics.

        Checks:
        - Game length distribution (detect abnormally short/long games)
        - Win rate balance across players
        - Board type diversity

        Args:
            db_path: Path to SQLite database

        Returns:
            True if quality metrics are acceptable, False otherwise
        """
        db_path = Path(db_path)

        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            return False

        issues = []

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()

            # Analyze game length distribution by board type
            cursor.execute(
                """
                SELECT board_type,
                       MIN(total_moves) as min_moves,
                       MAX(total_moves) as max_moves,
                       AVG(total_moves) as avg_moves,
                       COUNT(*) as count
                FROM games
                WHERE winner IS NOT NULL
                GROUP BY board_type
                """
            )

            for row in cursor.fetchall():
                board_type, min_moves, max_moves, avg_moves, count = row

                # Check against expected ranges
                expected_range = self.GAME_LENGTH_RANGES.get(
                    board_type, (10, 1000)
                )

                if min_moves < expected_range[0] / 2:
                    issues.append(
                        f"{board_type}: suspiciously short games (min={min_moves})"
                    )

                if max_moves > expected_range[1] * 2:
                    issues.append(
                        f"{board_type}: suspiciously long games (max={max_moves})"
                    )

            # Check win rate balance (2-player games only)
            cursor.execute(
                """
                SELECT winner, COUNT(*) as wins
                FROM games
                WHERE winner IS NOT NULL AND num_players = 2
                GROUP BY winner
                """
            )

            win_counts = {}
            for row in cursor.fetchall():
                winner, count = row
                win_counts[winner] = count

            if len(win_counts) == 2:
                total_wins = sum(win_counts.values())
                for player, wins in win_counts.items():
                    win_rate = wins / total_wins

                    # Flag if win rate is too skewed (< 30% or > 70%)
                    if win_rate < 0.30 or win_rate > 0.70:
                        issues.append(
                            f"Player {player} win rate imbalanced: {win_rate:.1%}"
                        )

            conn.close()

            if issues:
                for issue in issues:
                    logger.warning(f"Quality issue: {issue}")
                return False

            logger.info(f"Game quality checks passed for {db_path}")
            return True

        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return False

    def get_quality_score(self, db_path: str | Path) -> float:
        """Compute overall quality score for database.

        Combines results from schema, integrity, and quality checks
        into a single score from 0.0 (worst) to 1.0 (perfect).

        Args:
            db_path: Path to SQLite database

        Returns:
            Quality score between 0.0 and 1.0
        """
        db_path = Path(db_path)

        if not db_path.exists():
            return 0.0

        # Generate comprehensive report
        report = self._generate_full_report(db_path)
        self.last_report = report

        return report.quality_score

    def _generate_full_report(self, db_path: Path) -> DataQualityReport:
        """Generate comprehensive quality report.

        Args:
            db_path: Path to database

        Returns:
            DataQualityReport with all findings
        """
        report = DataQualityReport(database_path=str(db_path))

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()

            # Get total and valid game counts
            cursor.execute("SELECT COUNT(*) FROM games")
            report.total_games = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM games WHERE winner IS NOT NULL")
            report.valid_games = cursor.fetchone()[0]

            # Component scores (0-100 each)
            schema_score = 100 if self.check_schema_consistency(db_path) else 0
            integrity_score = 100 if self.check_data_integrity(db_path) else 50
            quality_score = 100 if self.check_game_quality(db_path) else 70

            # CRITICAL: Check for games with move data (export will fail silently without this)
            moves_ok, moves_stats = self.check_games_with_moves(db_path)
            if moves_stats["issue"]:
                report.issues.append(moves_stats["issue"])
            if not moves_ok:
                if moves_stats["coverage_percent"] < 10.0:
                    report.recommendations.append(
                        "CRITICAL: Database has almost no move data! "
                        "Re-run selfplay with proper move recording, or fix JSONL-to-DB import."
                    )
                else:
                    report.recommendations.append(
                        "Some games are missing move data. Check data import pipeline."
                    )

            # Score based on move coverage (0-100)
            moves_score = int(moves_stats["coverage_percent"])
            report.metadata["games_with_moves"] = moves_stats["games_with_moves"]
            report.metadata["games_without_moves"] = moves_stats["games_without_moves"]
            report.metadata["move_coverage_percent"] = moves_stats["coverage_percent"]

            # Completeness score
            completeness_score = 100
            if report.total_games > 0:
                completeness_ratio = report.valid_games / report.total_games
                completeness_score = int(completeness_ratio * 100)

                if completeness_ratio < 0.95:
                    report.issues.append(
                        f"Low completion rate: {completeness_ratio:.1%}"
                    )
                    report.recommendations.append(
                        "Investigate incomplete games and improve game completion rate"
                    )

            # Overall quality score (weighted average)
            # Move coverage is given high weight because exports fail without it
            weights = {
                "schema": 0.15,
                "integrity": 0.25,
                "quality": 0.20,
                "completeness": 0.15,
                "moves": 0.25,  # High weight - critical for training export
            }

            overall = (
                schema_score * weights["schema"] +
                integrity_score * weights["integrity"] +
                quality_score * weights["quality"] +
                completeness_score * weights["completeness"] +
                moves_score * weights["moves"]
            ) / 100.0

            report.quality_score = overall

            # Add metadata
            cursor.execute(
                """
                SELECT board_type, num_players, COUNT(*) as count
                FROM games WHERE winner IS NOT NULL
                GROUP BY board_type, num_players
                ORDER BY count DESC
                """
            )

            config_counts = {}
            for row in cursor.fetchall():
                board_type, num_players, count = row
                config_counts[f"{board_type}_{num_players}p"] = count

            report.metadata["config_breakdown"] = config_counts
            report.metadata["schema_valid"] = schema_score == 100
            report.metadata["integrity_valid"] = integrity_score == 100
            report.metadata["quality_valid"] = quality_score == 100
            report.metadata["moves_valid"] = moves_ok

            # Generate recommendations based on score
            if report.quality_score < 0.5:
                report.recommendations.append(
                    "Database quality is poor - consider regenerating data"
                )
            elif report.quality_score < 0.8:
                report.recommendations.append(
                    "Database quality is acceptable but could be improved"
                )
            else:
                report.recommendations.append(
                    "Database quality is good - safe to use for training"
                )

            conn.close()

            # Phase 3 Feedback Loop: Emit quality events for selfplay throttling
            self._emit_quality_events(report)

        except Exception as e:
            report.issues.append(f"Failed to generate report: {e}")
            report.quality_score = 0.0

        return report

    def _emit_quality_events(self, report: DataQualityReport) -> None:
        """Emit quality-related events for the feedback loop.

        December 2025: Phase 3 of self-improvement feedback loop.
        Emits events that selfplay_runner can subscribe to for throttling.

        Dec 27, 2025: Fixed to work in both sync and async contexts.
        Previously used asyncio.run() which failed silently in daemon contexts
        (where an event loop is already running).

        Args:
            report: Quality report with score and issues
        """
        try:
            import asyncio
            from app.coordination.event_router import DataEventType, emit_data_event

            # Extract config info from metadata if available
            config_breakdown = report.metadata.get("config_breakdown", {})
            board_type = "unknown"
            num_players = 2

            # Try to extract from first config
            for config_key in config_breakdown:
                parts = config_key.rsplit("_", 1)
                if len(parts) == 2:
                    board_type = parts[0]
                    try:
                        num_players = int(parts[1].replace("p", ""))
                    except ValueError:
                        pass
                    break

            # Helper to emit events in both sync and async contexts
            def _emit_event_sync(coro):
                """Execute event emission, handling both sync and async contexts."""
                try:
                    # Check if we're in an async context
                    loop = asyncio.get_running_loop()
                    # In async context: schedule as fire-and-forget task
                    loop.create_task(coro)
                except RuntimeError:
                    # No running loop: create new one
                    asyncio.run(coro)

            # Always emit QUALITY_SCORE_UPDATED
            quality_payload = {
                "database_path": report.database_path,
                "board_type": board_type,
                "num_players": num_players,
                "quality_score": report.quality_score,
                "total_games": report.total_games,
                "valid_games": report.valid_games,
                "issues_count": len(report.issues),
                "issues": report.issues[:5],  # First 5 issues
            }
            _emit_event_sync(
                emit_data_event(
                    event_type=DataEventType.QUALITY_SCORE_UPDATED,
                    payload=quality_payload,
                )
            )
            logger.debug(
                f"[DataQuality] Emitted QUALITY_SCORE_UPDATED: "
                f"score={report.quality_score:.2f}, db={report.database_path}"
            )

            # Emit LOW_QUALITY_DATA_WARNING if score drops below threshold
            QUALITY_THRESHOLD = 0.8
            if report.quality_score < QUALITY_THRESHOLD:
                warning_payload = {
                    "database_path": report.database_path,
                    "board_type": board_type,
                    "num_players": num_players,
                    "quality_score": report.quality_score,
                    "threshold": QUALITY_THRESHOLD,
                    "severity": "critical" if report.quality_score < 0.5 else "warning",
                    "issues": report.issues,
                    "recommendations": report.recommendations,
                }
                _emit_event_sync(
                    emit_data_event(
                        event_type=DataEventType.LOW_QUALITY_DATA_WARNING,
                        payload=warning_payload,
                    )
                )
                logger.warning(
                    f"[DataQuality] Emitted LOW_QUALITY_DATA_WARNING: "
                    f"score={report.quality_score:.2f} < threshold={QUALITY_THRESHOLD}"
                )

        except ImportError:
            logger.debug("[DataQuality] Event router not available, skipping event emission")
        except Exception as e:
            logger.debug(f"[DataQuality] Failed to emit quality events: {e}")


# =============================================================================
# Pre-Export Validation
# =============================================================================


def validate_database_for_export(
    db_path: str | Path,
    board_type: str | None = None,
    num_players: int | None = None,
    min_coverage_percent: float = 10.0,
    warn_coverage_percent: float = 90.0,
) -> tuple[bool, str]:
    """Validate a database is suitable for training data export.

    This is a convenience function for the export pipeline that checks
    the most critical issues that would cause silent export failures.

    Args:
        db_path: Path to SQLite database
        board_type: Optional board type filter
        num_players: Optional player count filter
        min_coverage_percent: Fail if move coverage is below this (default 10%)
        warn_coverage_percent: Warn if move coverage is below this (default 90%)

    Returns:
        Tuple of (is_valid, message) where:
        - is_valid: True if database is suitable for export
        - message: Description of validation result

    Example:
        valid, msg = validate_database_for_export("selfplay.db", "square8", 2)
        if not valid:
            print(f"Export blocked: {msg}")
            sys.exit(1)
    """
    db_path = Path(db_path)

    if not db_path.exists():
        return False, f"Database not found: {db_path}"

    checker = DatabaseQualityChecker()

    # Check for games with move data (critical for export)
    moves_ok, moves_stats = checker.check_games_with_moves(db_path)

    if moves_stats["total_games"] == 0:
        return False, "Database contains no games"

    coverage = moves_stats["coverage_percent"]
    games_with = moves_stats["games_with_moves"]
    total = moves_stats["total_games"]

    if coverage < min_coverage_percent:
        return False, (
            f"CRITICAL: Only {coverage:.1f}% of games have move data "
            f"({games_with}/{total}). Export would produce zero samples. "
            f"This usually means JSONL-to-DB import failed to populate game_moves table."
        )

    # Check for games matching the requested config
    if board_type and num_players:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*)
                FROM games g
                WHERE g.board_type = ?
                  AND g.num_players = ?
                  AND EXISTS (SELECT 1 FROM game_moves m WHERE m.game_id = g.game_id)
            """, (board_type, num_players))
            matching_games = cursor.fetchone()[0]
            conn.close()

            if matching_games == 0:
                return False, (
                    f"No {board_type} {num_players}p games with move data found. "
                    f"Database has {total} total games but none match this config."
                )
        except Exception as e:
            return False, f"Failed to query database: {e}"

    # Warn if coverage is low but above threshold
    if coverage < warn_coverage_percent:
        return True, (
            f"WARNING: Only {coverage:.1f}% of games have move data "
            f"({games_with}/{total}). Export will proceed but some games will be skipped."
        )

    # Dec 29, 2025: Additional validation for multiplayer games (3p/4p)
    # Check for corrupted move data (e.g., PLACE_RING moves with to=None)
    # This caused hex8_4p Elo regression from 1500 to 594
    if num_players and num_players >= 3:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
            cursor = conn.cursor()
            # Sample a few moves to check for corruption
            cursor.execute("""
                SELECT m.move_data
                FROM game_moves m
                JOIN games g ON m.game_id = g.game_id
                WHERE g.num_players = ?
                  AND g.board_type = ?
                LIMIT 100
            """, (num_players, board_type))
            rows = cursor.fetchall()
            conn.close()

            corrupted_count = 0
            for row in rows:
                try:
                    import json
                    move_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    # Check for PLACE_RING moves with to=None
                    if isinstance(move_data, dict):
                        move_type = move_data.get("type") or move_data.get("moveType")
                        if move_type in ("PLACE_RING", "place_ring"):
                            to_field = move_data.get("to")
                            if to_field is None:
                                corrupted_count += 1
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

            if corrupted_count > 0:
                corruption_rate = corrupted_count / max(len(rows), 1) * 100
                if corruption_rate > 50:
                    return False, (
                        f"CRITICAL: {corrupted_count}/{len(rows)} sampled moves ({corruption_rate:.0f}%) "
                        f"have corrupted data (PLACE_RING with to=None). "
                        f"This {num_players}p database needs regeneration."
                    )
                elif corruption_rate > 10:
                    return True, (
                        f"WARNING: {corrupted_count}/{len(rows)} sampled moves have corrupted data. "
                        f"Export will proceed but some games may fail replay."
                    )
        except Exception as e:
            # Don't block export on validation failure, just warn
            logger.warning(f"Multiplayer move validation failed: {e}")

    return True, f"Database OK: {games_with}/{total} games ({coverage:.1f}%) have move data"


# =============================================================================
# Training Data Validator (NPZ)
# =============================================================================


class TrainingDataValidator:
    """Validates NPZ training data files.

    Performs comprehensive validation of NPZ feature/label quality:
    - Structure validation (required arrays present)
    - Data type validation
    - Feature distribution analysis
    - Outlier detection
    - Label validity checks

    Example:
        validator = TrainingDataValidator()

        # Basic validation
        valid = validator.validate_npz_file("batch_001.npz")

        # Detailed analysis
        stats = validator.check_feature_distribution("batch_001.npz")
        outliers = validator.detect_outliers("batch_001.npz")

        # Label validation
        label_ok = validator.validate_labels("batch_001.npz")
    """

    # Expected arrays in NPZ file
    REQUIRED_ARRAYS = ["features", "values"]
    OPTIONAL_ARRAYS = ["policy_indices", "policy_values", "globals"]

    # Value ranges for validation
    VALUE_RANGE = (-1.0, 1.0)
    POLICY_SUM_TOLERANCE = 0.01

    def __init__(self):
        self.last_stats: dict[str, Any] = {}

    def validate_npz_file(self, npz_path: str | Path) -> bool:
        """Validate NPZ structure and data types.

        Args:
            npz_path: Path to NPZ file

        Returns:
            True if validation passes, False otherwise
        """
        npz_path = Path(npz_path)

        if not npz_path.exists():
            logger.error(f"NPZ file not found: {npz_path}")
            return False

        try:
            data = safe_load_npz(npz_path)

            # Check required arrays
            available = set(data.keys())
            required = set(self.REQUIRED_ARRAYS)

            missing = required - available
            if missing:
                logger.error(f"Missing required arrays: {missing}")
                return False

            # Validate array shapes and types
            features = data["features"]
            values = data["values"]

            # Features should be 4D: (N, C, H, W)
            if features.ndim != 4:
                logger.error(f"Features should be 4D, got {features.ndim}D")
                return False

            # Values should be 1D: (N,)
            if values.ndim != 1:
                logger.error(f"Values should be 1D, got {values.ndim}D")
                return False

            # Sample counts should match
            if len(features) != len(values):
                logger.error(
                    f"Sample count mismatch: features={len(features)}, "
                    f"values={len(values)}"
                )
                return False

            # Check for policy data consistency
            if "policy_indices" in available:
                if "policy_values" not in available:
                    logger.error("policy_indices present but policy_values missing")
                    return False

                policy_indices = data["policy_indices"]
                policy_values = data["policy_values"]

                if len(policy_indices) != len(features):
                    logger.error("Policy data length mismatch")
                    return False

            logger.info(f"NPZ validation passed: {len(features)} samples")
            return True

        except Exception as e:
            logger.error(f"NPZ validation failed: {e}")
            return False

    def check_feature_distribution(self, npz_path: str | Path) -> dict[str, Any]:
        """Analyze feature statistics.

        Computes distribution statistics for feature arrays:
        - Mean, std, min, max per channel
        - NaN/Inf detection
        - Sparsity analysis

        Args:
            npz_path: Path to NPZ file

        Returns:
            Dictionary with feature statistics
        """
        npz_path = Path(npz_path)
        stats = {}

        if not npz_path.exists():
            logger.error(f"NPZ file not found: {npz_path}")
            return stats

        try:
            data = safe_load_npz(npz_path)
            features = data["features"]

            # Overall statistics
            stats["shape"] = features.shape
            stats["dtype"] = str(features.dtype)
            stats["num_samples"] = len(features)
            stats["num_channels"] = features.shape[1]

            # Per-channel statistics
            channel_stats = []
            for c in range(features.shape[1]):
                channel = features[:, c, :, :]

                ch_stats = {
                    "channel": c,
                    "mean": float(np.mean(channel)),
                    "std": float(np.std(channel)),
                    "min": float(np.min(channel)),
                    "max": float(np.max(channel)),
                    "has_nan": bool(np.any(np.isnan(channel))),
                    "has_inf": bool(np.any(np.isinf(channel))),
                    "sparsity": float(np.mean(channel == 0.0)),
                }
                channel_stats.append(ch_stats)

            stats["channels"] = channel_stats

            # Check for problematic channels
            nan_channels = [cs["channel"] for cs in channel_stats if cs["has_nan"]]
            inf_channels = [cs["channel"] for cs in channel_stats if cs["has_inf"]]

            if nan_channels:
                stats["warning"] = f"NaN values in channels: {nan_channels}"
            if inf_channels:
                stats["warning"] = f"Inf values in channels: {inf_channels}"

            self.last_stats = stats
            logger.info(f"Feature distribution analysis complete: {npz_path}")

        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            stats["error"] = str(e)

        return stats

    def detect_outliers(self, npz_path: str | Path, threshold: float = 3.0) -> dict[str, Any]:
        """Detect anomalous samples using z-score.

        Args:
            npz_path: Path to NPZ file
            threshold: Z-score threshold for outlier detection

        Returns:
            Dictionary with outlier information
        """
        npz_path = Path(npz_path)
        outliers = {"outlier_indices": [], "num_outliers": 0}

        if not npz_path.exists():
            logger.error(f"NPZ file not found: {npz_path}")
            return outliers

        try:
            data = safe_load_npz(npz_path)
            features = data["features"]
            values = data["values"]

            # Compute feature norms per sample
            norms = np.linalg.norm(
                features.reshape(len(features), -1), axis=1
            )

            # Z-score based outlier detection
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)

            if std_norm > 0:
                z_scores = np.abs((norms - mean_norm) / std_norm)
                outlier_mask = z_scores > threshold
                outlier_indices = np.where(outlier_mask)[0].tolist()

                outliers["outlier_indices"] = outlier_indices
                outliers["num_outliers"] = len(outlier_indices)
                outliers["threshold"] = threshold
                outliers["mean_norm"] = float(mean_norm)
                outliers["std_norm"] = float(std_norm)

                if outlier_indices:
                    logger.warning(
                        f"Found {len(outlier_indices)} outliers "
                        f"(threshold={threshold})"
                    )
                else:
                    logger.info("No outliers detected")

        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            outliers["error"] = str(e)

        return outliers

    def validate_labels(self, npz_path: str | Path) -> bool:
        """Check value and policy label validity.

        Validates:
        - Values in valid range [-1, 1]
        - Policy probabilities sum to 1 (within tolerance)
        - No negative probabilities
        - No NaN/Inf in labels

        Args:
            npz_path: Path to NPZ file

        Returns:
            True if labels are valid, False otherwise
        """
        npz_path = Path(npz_path)

        if not npz_path.exists():
            logger.error(f"NPZ file not found: {npz_path}")
            return False

        issues = []

        try:
            data = safe_load_npz(npz_path)
            values = data["values"]

            # Validate value range
            out_of_range = np.sum(
                (values < self.VALUE_RANGE[0]) | (values > self.VALUE_RANGE[1])
            )
            if out_of_range > 0:
                issues.append(
                    f"{out_of_range} values out of range {self.VALUE_RANGE}"
                )

            # Check for NaN/Inf in values
            if np.any(np.isnan(values)):
                issues.append("NaN values found in value labels")
            if np.any(np.isinf(values)):
                issues.append("Inf values found in value labels")

            # Validate policy if present
            if "policy_indices" in data and "policy_values" in data:
                policy_indices = data["policy_indices"]
                policy_values = data["policy_values"]

                # Check a sample of policies
                num_samples = min(1000, len(policy_indices))
                sample_indices = np.random.choice(
                    len(policy_indices), num_samples, replace=False
                )

                for idx in sample_indices:
                    p_vals = policy_values[idx]

                    # Skip empty policies (terminal states)
                    if len(p_vals) == 0:
                        continue

                    # Check for negative probabilities
                    if np.any(p_vals < 0):
                        issues.append(
                            f"Negative probabilities in sample {idx}"
                        )

                    # Check sum to 1
                    p_sum = np.sum(p_vals)
                    if abs(p_sum - 1.0) > self.POLICY_SUM_TOLERANCE:
                        issues.append(
                            f"Policy sum {p_sum:.4f} != 1.0 in sample {idx}"
                        )

            if issues:
                for issue in issues:
                    logger.warning(f"Label validation issue: {issue}")
                return False

            logger.info(f"Label validation passed for {npz_path}")
            return True

        except Exception as e:
            logger.error(f"Label validation failed: {e}")
            return False


# =============================================================================
# CLI Interface
# =============================================================================


def scan_all_databases() -> None:
    """Scan all discovered databases and report quality."""
    try:
        from app.utils.game_discovery import GameDiscovery

        discovery = GameDiscovery()
        databases = discovery.find_all_databases()

        if not databases:
            print("No databases found")
            return

        print(f"\nScanning {len(databases)} databases...\n")

        checker = DatabaseQualityChecker()
        results = []

        for db_info in databases:
            print(f"Checking {db_info.path.name}...", end=" ")
            score = checker.get_quality_score(db_info.path)
            results.append((db_info.path, score, checker.last_report))

            # Color-coded output
            if score >= 0.8:
                status = "GOOD"
            elif score >= 0.5:
                status = "OK"
            else:
                status = "POOR"

            print(f"{status} ({score:.1%})")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        avg_score = sum(r[1] for r in results) / len(results)
        print(f"Average quality score: {avg_score:.1%}")
        print(f"Total databases scanned: {len(results)}")

        # Category breakdown
        good_dbs = [r for r in results if r[1] >= 0.8]
        ok_dbs = [r for r in results if 0.5 <= r[1] < 0.8]
        poor_dbs = [r for r in results if r[1] < 0.5]

        print(f"  - Good (>=80%): {len(good_dbs)}")
        print(f"  - Acceptable (50-80%): {len(ok_dbs)}")
        print(f"  - Poor (<50%): {len(poor_dbs)}")

        if poor_dbs:
            print(f"\nDatabases needing attention ({len(poor_dbs)}):")
            for db_path, score, report in poor_dbs:
                print(f"  - {db_path.name}: {score:.1%}")
                if report and report.issues:
                    for issue in report.issues[:3]:  # Show first 3 issues
                        print(f"    * {issue}")

        # Total games summary
        total_games = sum(r[2].total_games for r in results if r[2])
        total_valid = sum(r[2].valid_games for r in results if r[2])
        if total_games > 0:
            print(f"\nTotal games across all databases: {total_games:,}")
            print(f"Valid games: {total_valid:,} ({total_valid/total_games:.1%})")

    except ImportError:
        logger.error("GameDiscovery not available - cannot scan all databases")
        sys.exit(1)


# =============================================================================
# Multiplayer Training Data Validation (December 2025)
# =============================================================================


@dataclass
class MultiplayerValidationResult:
    """Result of multiplayer training data validation."""

    valid: bool
    num_samples: int
    expected_players: int
    values_mp_shape: tuple[int, ...] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        lines = [
            f"Multiplayer Validation: {status}",
            f"  Samples: {self.num_samples:,}",
            f"  Expected players: {self.expected_players}",
        ]
        if self.values_mp_shape:
            lines.append(f"  values_mp shape: {self.values_mp_shape}")
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        if self.warnings:
            lines.append("  Warnings:")
            for warn in self.warnings:
                lines.append(f"    - {warn}")
        return "\n".join(lines)


def validate_multiplayer_training_data(
    npz_path: str | Path,
    expected_players: int,
    min_samples: int = 1000,
) -> MultiplayerValidationResult:
    """Validate training data for multiplayer model training.

    This function performs critical validations to prevent training failures
    caused by data corruption issues that led to hex8_4p and square19_3p
    Elo regressions (594 and 409 from baseline 1500).

    Checks:
    1. values_mp array dimension matches expected player count
    2. Minimum sample count for statistical significance
    3. values_mp values are in valid range [-1, 1]
    4. num_players array matches expected player count

    Args:
        npz_path: Path to NPZ training file
        expected_players: Expected number of players (2, 3, or 4)
        min_samples: Minimum required samples (default 1000)

    Returns:
        MultiplayerValidationResult with validation status and details

    Raises:
        ValueError: If npz_path doesn't exist

    Example:
        result = validate_multiplayer_training_data(
            "data/training/hex8_4p.npz",
            expected_players=4,
            min_samples=5000,
        )
        if not result.valid:
            raise ValueError(f"Training data validation failed: {result}")
    """
    npz_path = Path(npz_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not npz_path.exists():
        return MultiplayerValidationResult(
            valid=False,
            num_samples=0,
            expected_players=expected_players,
            values_mp_shape=None,
            errors=[f"NPZ file not found: {npz_path}"],
        )

    try:
        data = safe_load_npz(npz_path)
    except Exception as e:
        return MultiplayerValidationResult(
            valid=False,
            num_samples=0,
            expected_players=expected_players,
            values_mp_shape=None,
            errors=[f"Failed to load NPZ: {e}"],
        )

    # Check for required multiplayer arrays
    if "values_mp" not in data:
        return MultiplayerValidationResult(
            valid=False,
            num_samples=len(data.get("features", [])),
            expected_players=expected_players,
            values_mp_shape=None,
            errors=[
                "Missing 'values_mp' array - required for multiplayer training. "
                "Regenerate data with export_replay_dataset.py --multi-player-values"
            ],
        )

    values_mp = data["values_mp"]
    num_samples = len(values_mp)
    values_mp_shape = values_mp.shape

    # Check 1: Sample count
    if num_samples < min_samples:
        errors.append(
            f"Insufficient samples: {num_samples:,} < {min_samples:,} minimum. "
            f"Need more games for statistically significant training."
        )

    # Check 2: values_mp dimension matches expected players
    if len(values_mp_shape) != 2:
        errors.append(
            f"values_mp should be 2D (samples, players), got {len(values_mp_shape)}D"
        )
    else:
        max_players_dim = values_mp_shape[1]
        if max_players_dim < expected_players:
            errors.append(
                f"values_mp dimension {max_players_dim} < expected {expected_players} players. "
                f"Data was generated for fewer players than training config."
            )

    # Check 3: num_players array consistency
    if "num_players" in data:
        num_players_arr = data["num_players"]

        # All samples should have same player count
        unique_players = np.unique(num_players_arr)
        if len(unique_players) > 1:
            warnings.append(
                f"Mixed player counts in data: {unique_players.tolist()}. "
                f"Training may be suboptimal with inconsistent data."
            )

        # Check most common matches expected
        if len(unique_players) == 1 and unique_players[0] != expected_players:
            errors.append(
                f"Data generated for {unique_players[0]}p but training expects {expected_players}p. "
                f"Use correct training data or regenerate with --num-players {expected_players}."
            )

        # Check for invalid player counts
        invalid_mask = (num_players_arr < 2) | (num_players_arr > 4)
        if np.any(invalid_mask):
            invalid_count = np.sum(invalid_mask)
            errors.append(
                f"{invalid_count:,} samples have invalid player count (not in 2-4 range)"
            )
    else:
        warnings.append(
            "No 'num_players' array - cannot verify per-sample player count consistency"
        )

    # Check 4: values_mp value range (should be in [-1, 1])
    if len(values_mp_shape) == 2:
        # Only check active player columns (first expected_players columns)
        active_values = values_mp[:, :expected_players]

        value_min = float(np.min(active_values))
        value_max = float(np.max(active_values))

        if value_min < -1.0 or value_max > 1.0:
            errors.append(
                f"values_mp out of range: [{value_min:.4f}, {value_max:.4f}] "
                f"(expected [-1, 1])"
            )

        # Check for all-zero values (common corruption sign)
        zero_rows = np.all(active_values == 0, axis=1)
        zero_count = int(np.sum(zero_rows))
        if zero_count > 0:
            zero_pct = zero_count / num_samples * 100
            if zero_pct > 10:
                errors.append(
                    f"{zero_count:,} samples ({zero_pct:.1f}%) have all-zero values "
                    f"(likely missing winner data)"
                )
            elif zero_pct > 1:
                warnings.append(
                    f"{zero_count:,} samples ({zero_pct:.1f}%) have all-zero values"
                )

    # Check 5: Feature/value array length consistency
    if "features" in data:
        features_len = len(data["features"])
        if features_len != num_samples:
            errors.append(
                f"Sample count mismatch: features={features_len:,}, "
                f"values_mp={num_samples:,}"
            )

    valid = len(errors) == 0

    return MultiplayerValidationResult(
        valid=valid,
        num_samples=num_samples,
        expected_players=expected_players,
        values_mp_shape=values_mp_shape,
        errors=errors,
        warnings=warnings,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Quality Tools for RingRift Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a specific database
  python -m app.training.data_quality --db data/games/selfplay.db

  # Validate NPZ training data
  python -m app.training.data_quality --npz data/training/batch_001.npz

  # Scan all discovered databases
  python -m app.training.data_quality --all

  # Detailed NPZ analysis
  python -m app.training.data_quality --npz data/training/batch_001.npz --detailed
        """
    )

    parser.add_argument(
        "--db",
        type=str,
        help="Path to database file to check"
    )
    parser.add_argument(
        "--npz",
        type=str,
        help="Path to NPZ file to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all discovered databases"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics (for NPZ validation)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify data checksums in NPZ file (requires --npz)"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Execute requested operation
    if args.all:
        scan_all_databases()

    elif args.db:
        checker = DatabaseQualityChecker()
        score = checker.get_quality_score(args.db)

        if checker.last_report:
            print(checker.last_report)
        else:
            print(f"Quality score: {score:.1%}")

    elif args.npz:
        validator = TrainingDataValidator()

        print(f"\nValidating {args.npz}...")
        print("-" * 70)

        # Basic validation
        valid = validator.validate_npz_file(args.npz)
        print(f"Structure valid: {valid}")

        # Checksum verification (December 2025)
        checksums_valid = True
        if args.verify_checksums:
            checksum_ok, computed, errors = verify_npz_checksums(args.npz)
            checksums_valid = checksum_ok
            if checksum_ok:
                print(f"Checksums valid: True ({len(computed)} arrays verified)")
            else:
                print(f"Checksums valid: False")
                for err in errors:
                    print(f"  - {err}")
        else:
            # Check if checksums exist but skip verification
            try:
                with safe_load_npz(args.npz) as data:
                    has_checksums = "data_checksums" in data
                    if has_checksums:
                        print(f"Checksums present: Yes (use --verify-checksums to verify)")
                    else:
                        print(f"Checksums present: No (legacy export)")
            except (FileNotFoundError, OSError, ValueError, zipfile.BadZipFile):
                pass

        # Label validation
        labels_valid = validator.validate_labels(args.npz)
        print(f"Labels valid: {labels_valid}")

        # Outlier detection
        outliers = validator.detect_outliers(args.npz)
        print(f"Outliers detected: {outliers['num_outliers']}")

        if args.detailed:
            # Feature distribution
            print("\nFeature Distribution Analysis:")
            print("-" * 70)
            stats = validator.check_feature_distribution(args.npz)

            if "channels" in stats:
                for ch_stats in stats["channels"]:
                    print(
                        f"  Channel {ch_stats['channel']}: "
                        f"mean={ch_stats['mean']:.4f}, "
                        f"std={ch_stats['std']:.4f}, "
                        f"range=[{ch_stats['min']:.4f}, {ch_stats['max']:.4f}], "
                        f"sparsity={ch_stats['sparsity']:.2%}"
                    )

            if "warning" in stats:
                print(f"\nWarning: {stats['warning']}")

        print("-" * 70)

        all_valid = valid and labels_valid
        if args.verify_checksums:
            all_valid = all_valid and checksums_valid

        if all_valid:
            print("Overall: PASS")
            sys.exit(0)
        else:
            print("Overall: FAIL")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
