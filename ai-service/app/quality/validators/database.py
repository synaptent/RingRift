"""Database Validator.

Validates game database files for integrity and quality.

December 30, 2025: Created as part of Priority 3.4 consolidation effort.
Migrates validation logic from app/training/data_quality.py.

Usage:
    from app.quality.validators.database import DatabaseValidator

    validator = DatabaseValidator()
    result = validator.validate("data/games/selfplay.db")
    if not result.is_valid:
        print(f"Errors: {result.errors}")
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.quality.types import ValidationResult
from app.quality.validators.base import PathValidator, ValidatorConfig

__all__ = [
    "DatabaseValidator",
    "DatabaseValidatorConfig",
]

logger = logging.getLogger(__name__)


@dataclass
class DatabaseValidatorConfig(ValidatorConfig):
    """Configuration for database validation.

    Attributes:
        min_games: Minimum number of games required
        min_moves_per_game: Minimum average moves per game
        check_move_integrity: Whether to validate move data
        check_replay_integrity: Whether to validate replay capability
    """

    min_games: int = 0
    min_moves_per_game: int = 5
    check_move_integrity: bool = True
    check_replay_integrity: bool = False


class DatabaseValidator(PathValidator):
    """Validator for game database files.

    Checks:
    - Database file exists and is readable
    - Required tables present (games, moves)
    - Schema validity
    - Data integrity (game counts, move counts)
    - Optional: replay capability
    """

    VALIDATOR_NAME = "database"
    VALIDATOR_VERSION = "1.0.0"

    # Required tables for a valid game database
    # Note: moves table can be "moves" or "game_moves" depending on schema version
    REQUIRED_TABLES = {"games"}
    MOVE_TABLE_NAMES = {"moves", "game_moves"}  # Either is acceptable

    # Required columns in games table
    GAMES_COLUMNS = {"game_id", "board_type", "num_players"}

    # Required columns in moves table
    MOVES_COLUMNS = {"game_id", "move_number"}

    def __init__(self, config: DatabaseValidatorConfig | None = None):
        """Initialize the database validator.

        Args:
            config: Validator configuration
        """
        self._db_config = config or DatabaseValidatorConfig()
        self._moves_table: str | None = None
        super().__init__(config=self._db_config)

    def _validate_file(self, path: Path) -> ValidationResult:
        """Validate database file contents."""
        result = ValidationResult(is_valid=True)

        try:
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            # Check tables exist
            table_errors = self._check_tables(cursor)
            for error in table_errors:
                result.add_error(error)

            if result.is_valid:
                # Check schema
                schema_errors = self._check_schema(cursor)
                for error in schema_errors:
                    result.add_error(error)

            if result.is_valid:
                # Check data integrity
                integrity_errors, warnings = self._check_integrity(cursor)
                for error in integrity_errors:
                    result.add_error(error)
                for warning in warnings:
                    result.add_warning(warning)

            # Get metadata
            if result.is_valid:
                result.metadata.update(self._get_metadata(cursor))

            conn.close()

        except sqlite3.Error as e:
            result.add_error(f"SQLite error: {e}")

        except Exception as e:
            result.add_error(f"Validation error: {e}")

        return result

    def _check_tables(self, cursor: sqlite3.Cursor) -> list[str]:
        """Check required tables exist."""
        errors = []

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        missing = self.REQUIRED_TABLES - tables
        if missing:
            errors.append(f"Missing tables: {missing}")

        # Check for moves table (either name is acceptable)
        has_moves = bool(self.MOVE_TABLE_NAMES & tables)
        if not has_moves:
            errors.append(f"Missing moves table (expected one of: {self.MOVE_TABLE_NAMES})")

        # Store which moves table exists for later use
        self._moves_table = next(
            (t for t in self.MOVE_TABLE_NAMES if t in tables), None
        )

        return errors

    def _check_schema(self, cursor: sqlite3.Cursor) -> list[str]:
        """Check table schemas are valid."""
        errors = []

        # Check games table columns
        cursor.execute("PRAGMA table_info(games)")
        games_columns = {row[1] for row in cursor.fetchall()}
        missing_games = self.GAMES_COLUMNS - games_columns
        if missing_games:
            errors.append(f"Games table missing columns: {missing_games}")

        # Check moves table columns (use whichever table exists)
        if self._moves_table:
            cursor.execute(f"PRAGMA table_info({self._moves_table})")
            moves_columns = {row[1] for row in cursor.fetchall()}
            missing_moves = self.MOVES_COLUMNS - moves_columns
            if missing_moves:
                errors.append(f"Moves table missing columns: {missing_moves}")

        return errors

    def _check_integrity(
        self, cursor: sqlite3.Cursor
    ) -> tuple[list[str], list[str]]:
        """Check data integrity."""
        errors = []
        warnings = []

        # Count games
        cursor.execute("SELECT COUNT(*) FROM games")
        game_count = cursor.fetchone()[0]

        if game_count < self._db_config.min_games:
            errors.append(
                f"Insufficient games: {game_count} < {self._db_config.min_games}"
            )

        if game_count == 0:
            warnings.append("Database contains no games")
            return errors, warnings

        # Check moves (use whichever table exists)
        move_count = 0
        if self._moves_table:
            cursor.execute(f"SELECT COUNT(*) FROM {self._moves_table}")
            move_count = cursor.fetchone()[0]

            if move_count == 0:
                errors.append("No moves in database")
            else:
                avg_moves = move_count / game_count
                if avg_moves < self._db_config.min_moves_per_game:
                    warnings.append(
                        f"Low average moves per game: {avg_moves:.1f}"
                    )

            # Check for orphan moves (moves without games)
            if self._db_config.check_move_integrity and move_count > 0:
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT m.game_id)
                    FROM {self._moves_table} m
                    LEFT JOIN games g ON m.game_id = g.game_id
                    WHERE g.game_id IS NULL
                """)
                orphan_count = cursor.fetchone()[0]
                if orphan_count > 0:
                    warnings.append(f"Found {orphan_count} orphan move sets")

        return errors, warnings

    def _get_metadata(self, cursor: sqlite3.Cursor) -> dict[str, Any]:
        """Extract database metadata."""
        metadata: dict[str, Any] = {}

        # Game count
        cursor.execute("SELECT COUNT(*) FROM games")
        metadata["game_count"] = cursor.fetchone()[0]

        # Move count (use whichever table exists)
        if self._moves_table:
            cursor.execute(f"SELECT COUNT(*) FROM {self._moves_table}")
            metadata["move_count"] = cursor.fetchone()[0]
        else:
            metadata["move_count"] = 0

        # Board types
        cursor.execute("SELECT DISTINCT board_type FROM games")
        metadata["board_types"] = [row[0] for row in cursor.fetchall()]

        # Player counts
        cursor.execute("SELECT DISTINCT num_players FROM games")
        metadata["player_counts"] = [row[0] for row in cursor.fetchall()]

        return metadata

    def get_database_stats(self, path: str | Path) -> dict[str, Any]:
        """Get detailed database statistics.

        Args:
            path: Path to database file

        Returns:
            Dictionary with database statistics
        """
        path = Path(path)
        stats: dict[str, Any] = {"path": str(path), "valid": False}

        if not path.exists():
            stats["error"] = "File not found"
            return stats

        try:
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM games")
            stats["game_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM moves")
            stats["move_count"] = cursor.fetchone()[0]

            # Config distribution
            cursor.execute("""
                SELECT board_type, num_players, COUNT(*) as count
                FROM games
                GROUP BY board_type, num_players
                ORDER BY count DESC
            """)
            stats["config_distribution"] = [
                {"board_type": row[0], "num_players": row[1], "count": row[2]}
                for row in cursor.fetchall()
            ]

            # Winner distribution
            cursor.execute("""
                SELECT winner, COUNT(*) as count
                FROM games
                GROUP BY winner
            """)
            stats["winner_distribution"] = {
                row[0]: row[1] for row in cursor.fetchall()
            }

            stats["valid"] = True
            conn.close()

        except Exception as e:
            stats["error"] = str(e)

        return stats
