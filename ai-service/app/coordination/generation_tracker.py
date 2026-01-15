"""Generation Tracker for RingRift AI.

Tracks model generations and their lineage to demonstrate iterative improvement.
Populates the generation_tracking.db with training history.

Schema (already exists in data/generation_tracking.db):
- model_generations: Tracks each model version with parent lineage
- generation_tournaments: Records head-to-head comparison results
- elo_progression: Tracks Elo rating over time per generation

Usage:
    from app.coordination.generation_tracker import GenerationTracker, get_generation_tracker

    tracker = get_generation_tracker()

    # Record a new trained model
    gen_id = tracker.record_generation(
        model_path="models/canonical_hex8_2p_v3.pth",
        board_type="hex8",
        num_players=2,
        parent_generation_id=2,  # Previous version
        training_games=5000,
        training_samples=250000,
    )

    # Record tournament result between generations
    tracker.record_tournament(
        gen_a=3, gen_b=2,
        gen_a_wins=55, gen_b_wins=45,
        gen_a_elo=1520, gen_b_elo=1480,
    )

    # Get improvement report
    report = tracker.get_improvement_report()
    print(report)

January 2026 - Created as part of MVP launch preparation.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.coordination.singleton_mixin import SingletonMixin

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/generation_tracking.db")


@dataclass
class GenerationInfo:
    """Information about a model generation."""
    generation_id: int
    model_path: str
    board_type: str
    num_players: int
    parent_generation: int | None
    created_at: float
    training_games: int | None
    training_samples: int | None

    @property
    def version(self) -> int:
        """Extract version number from model path (e.g., v3 from canonical_hex8_2p_v3.pth)."""
        match = re.search(r'_v(\d+)\.pth$', self.model_path)
        return int(match.group(1)) if match else 1


@dataclass
class TournamentResult:
    """Result of a head-to-head tournament between generations."""
    gen_a: int
    gen_b: int
    gen_a_wins: int
    gen_b_wins: int
    draws: int
    total_games: int
    gen_a_elo: float | None
    gen_b_elo: float | None
    timestamp: float
    # January 2026: Extended tournament metadata
    harness_type: str = "gumbel_mcts"
    difficulty: float = 1.0
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class EloSnapshot:
    """Elo rating snapshot for a generation."""
    generation_id: int
    elo: float
    games_played: int
    timestamp: float


class GenerationTracker(SingletonMixin):
    """Tracks model generations and their improvement over time.

    This class populates the generation_tracking.db to demonstrate
    iterative self-improvement of neural network models.
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the generation tracker.

        Args:
            db_path: Path to the generation tracking database.
                     Defaults to data/generation_tracking.db.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._ensure_tables()
        logger.info(f"GenerationTracker initialized with database: {self.db_path}")

    def _ensure_tables(self) -> None:
        """Ensure all required tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # model_generations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_generations (
                    generation_id INTEGER PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    parent_generation INTEGER,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    training_games INTEGER,
                    training_samples INTEGER,
                    FOREIGN KEY (parent_generation) REFERENCES model_generations(generation_id)
                )
            """)

            # generation_tournaments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_tournaments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gen_a INTEGER NOT NULL,
                    gen_b INTEGER NOT NULL,
                    gen_a_wins INTEGER DEFAULT 0,
                    gen_b_wins INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    total_games INTEGER DEFAULT 0,
                    gen_a_elo REAL,
                    gen_b_elo REAL,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    harness_type TEXT DEFAULT 'gumbel_mcts',
                    difficulty REAL DEFAULT 1.0,
                    ci_lower REAL,
                    ci_upper REAL,
                    FOREIGN KEY (gen_a) REFERENCES model_generations(generation_id),
                    FOREIGN KEY (gen_b) REFERENCES model_generations(generation_id)
                )
            """)

            # January 2026: Migrate existing tables to add new columns
            self._migrate_tournament_schema(cursor)

            # elo_progression table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elo_progression (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id INTEGER NOT NULL,
                    elo REAL NOT NULL,
                    games_played INTEGER NOT NULL,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (generation_id) REFERENCES model_generations(generation_id)
                )
            """)

            conn.commit()

    def _migrate_tournament_schema(self, cursor: sqlite3.Cursor) -> None:
        """Migrate existing generation_tournaments table to add new columns.

        January 2026: Adds harness_type, difficulty, ci_lower, ci_upper columns
        for enhanced tournament metadata tracking.
        """
        # Get existing columns
        cursor.execute("PRAGMA table_info(generation_tournaments)")
        columns = {row[1] for row in cursor.fetchall()}

        # Add missing columns
        if "harness_type" not in columns:
            try:
                cursor.execute(
                    "ALTER TABLE generation_tournaments ADD COLUMN harness_type TEXT DEFAULT 'gumbel_mcts'"
                )
                logger.info("Added harness_type column to generation_tournaments")
            except sqlite3.OperationalError:
                pass  # Column might already exist

        if "difficulty" not in columns:
            try:
                cursor.execute(
                    "ALTER TABLE generation_tournaments ADD COLUMN difficulty REAL DEFAULT 1.0"
                )
                logger.info("Added difficulty column to generation_tournaments")
            except sqlite3.OperationalError:
                pass

        if "ci_lower" not in columns:
            try:
                cursor.execute(
                    "ALTER TABLE generation_tournaments ADD COLUMN ci_lower REAL"
                )
                logger.info("Added ci_lower column to generation_tournaments")
            except sqlite3.OperationalError:
                pass

        if "ci_upper" not in columns:
            try:
                cursor.execute(
                    "ALTER TABLE generation_tournaments ADD COLUMN ci_upper REAL"
                )
                logger.info("Added ci_upper column to generation_tournaments")
            except sqlite3.OperationalError:
                pass

    def record_generation(
        self,
        model_path: str | Path,
        board_type: str,
        num_players: int,
        parent_generation_id: int | None = None,
        training_games: int | None = None,
        training_samples: int | None = None,
    ) -> int:
        """Record a new model generation.

        Args:
            model_path: Path to the model checkpoint
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)
            parent_generation_id: ID of the parent generation (if any)
            training_games: Number of games used for training
            training_samples: Number of training samples used

        Returns:
            The generation ID of the newly recorded generation.
        """
        model_path_str = str(model_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_generations
                (model_path, board_type, num_players, parent_generation, training_games, training_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_path_str, board_type, num_players, parent_generation_id, training_games, training_samples))

            generation_id = cursor.lastrowid
            conn.commit()

        logger.info(
            f"Recorded generation {generation_id}: {board_type}_{num_players}p "
            f"(parent={parent_generation_id}, games={training_games}, samples={training_samples})"
        )
        return generation_id

    def record_tournament(
        self,
        gen_a: int,
        gen_b: int,
        gen_a_wins: int,
        gen_b_wins: int,
        draws: int = 0,
        gen_a_elo: float | None = None,
        gen_b_elo: float | None = None,
        harness_type: str = "gumbel_mcts",
        difficulty: float = 1.0,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
    ) -> int:
        """Record a tournament result between two generations.

        Args:
            gen_a: First generation ID
            gen_b: Second generation ID
            gen_a_wins: Number of wins for generation A
            gen_b_wins: Number of wins for generation B
            draws: Number of draws
            gen_a_elo: Elo rating of generation A after tournament
            gen_b_elo: Elo rating of generation B after tournament
            harness_type: Harness used for tournament (January 2026)
            difficulty: Difficulty level used (January 2026)
            ci_lower: Wilson CI lower bound (January 2026)
            ci_upper: Wilson CI upper bound (January 2026)

        Returns:
            The tournament record ID.
        """
        total_games = gen_a_wins + gen_b_wins + draws

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO generation_tournaments
                (gen_a, gen_b, gen_a_wins, gen_b_wins, draws, total_games,
                 gen_a_elo, gen_b_elo, harness_type, difficulty, ci_lower, ci_upper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (gen_a, gen_b, gen_a_wins, gen_b_wins, draws, total_games,
                  gen_a_elo, gen_b_elo, harness_type, difficulty, ci_lower, ci_upper))

            record_id = cursor.lastrowid
            conn.commit()

        logger.info(
            f"Recorded tournament: gen {gen_a} vs gen {gen_b} - "
            f"{gen_a_wins}:{gen_b_wins} ({draws} draws), harness={harness_type}"
        )
        return record_id

    def record_elo_snapshot(
        self,
        generation_id: int,
        elo: float,
        games_played: int,
    ) -> int:
        """Record an Elo rating snapshot for a generation.

        Args:
            generation_id: The generation ID
            elo: Current Elo rating
            games_played: Total games played

        Returns:
            The snapshot record ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO elo_progression (generation_id, elo, games_played)
                VALUES (?, ?, ?)
            """, (generation_id, elo, games_played))

            record_id = cursor.lastrowid
            conn.commit()

        logger.debug(f"Recorded Elo snapshot: gen {generation_id} = {elo} ({games_played} games)")
        return record_id

    def get_generation(self, generation_id: int) -> GenerationInfo | None:
        """Get information about a specific generation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT generation_id, model_path, board_type, num_players,
                       parent_generation, created_at, training_games, training_samples
                FROM model_generations
                WHERE generation_id = ?
            """, (generation_id,))

            row = cursor.fetchone()
            if row:
                return GenerationInfo(*row)
            return None

    def get_latest_generation(self, board_type: str, num_players: int) -> GenerationInfo | None:
        """Get the latest generation for a board configuration."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT generation_id, model_path, board_type, num_players,
                       parent_generation, created_at, training_games, training_samples
                FROM model_generations
                WHERE board_type = ? AND num_players = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (board_type, num_players))

            row = cursor.fetchone()
            if row:
                return GenerationInfo(*row)
            return None

    def get_all_generations(self, board_type: str | None = None, num_players: int | None = None) -> list[GenerationInfo]:
        """Get all generations, optionally filtered by board type and players."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT generation_id, model_path, board_type, num_players,
                       parent_generation, created_at, training_games, training_samples
                FROM model_generations
            """
            params: list[Any] = []

            conditions = []
            if board_type:
                conditions.append("board_type = ?")
                params.append(board_type)
            if num_players:
                conditions.append("num_players = ?")
                params.append(num_players)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at ASC"

            cursor.execute(query, params)
            return [GenerationInfo(*row) for row in cursor.fetchall()]

    def get_lineage(self, generation_id: int) -> list[GenerationInfo]:
        """Get the full lineage (ancestors) of a generation."""
        lineage = []
        current_id: int | None = generation_id

        while current_id is not None:
            gen = self.get_generation(current_id)
            if gen:
                lineage.append(gen)
                current_id = gen.parent_generation
            else:
                break

        return lineage

    def get_tournaments_for_generation(self, generation_id: int) -> list[TournamentResult]:
        """Get all tournaments involving a specific generation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT gen_a, gen_b, gen_a_wins, gen_b_wins, draws,
                       total_games, gen_a_elo, gen_b_elo, timestamp,
                       COALESCE(harness_type, 'gumbel_mcts'),
                       COALESCE(difficulty, 1.0),
                       ci_lower, ci_upper
                FROM generation_tournaments
                WHERE gen_a = ? OR gen_b = ?
                ORDER BY timestamp DESC
            """, (generation_id, generation_id))

            return [TournamentResult(*row) for row in cursor.fetchall()]

    def get_elo_history(self, generation_id: int) -> list[EloSnapshot]:
        """Get Elo rating history for a generation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT generation_id, elo, games_played, timestamp
                FROM elo_progression
                WHERE generation_id = ?
                ORDER BY timestamp ASC
            """, (generation_id,))

            return [EloSnapshot(*row) for row in cursor.fetchall()]

    def get_improvement_report(self, board_type: str | None = None) -> str:
        """Generate a human-readable improvement report.

        Args:
            board_type: Optional filter for specific board type

        Returns:
            Formatted string showing improvement across generations.
        """
        lines = ["=== RingRift Generation Progress Report ===", ""]

        # Get all generations grouped by config
        generations = self.get_all_generations(board_type=board_type)

        if not generations:
            return "No generations recorded yet. Train models to see progress!"

        # Group by config
        by_config: dict[str, list[GenerationInfo]] = {}
        for gen in generations:
            config_key = f"{gen.board_type}_{gen.num_players}p"
            if config_key not in by_config:
                by_config[config_key] = []
            by_config[config_key].append(gen)

        for config_key, gens in sorted(by_config.items()):
            lines.append(f"### {config_key} ###")

            if len(gens) < 2:
                lines.append(f"  {len(gens)} generation(s) - need 2+ for comparison")
                lines.append("")
                continue

            # Get Elo for first and last generation
            first_gen = gens[0]
            last_gen = gens[-1]

            first_elo_history = self.get_elo_history(first_gen.generation_id)
            last_elo_history = self.get_elo_history(last_gen.generation_id)

            first_elo = first_elo_history[-1].elo if first_elo_history else None
            last_elo = last_elo_history[-1].elo if last_elo_history else None

            lines.append(f"  Generations: {len(gens)}")

            if first_elo is not None and last_elo is not None:
                delta = last_elo - first_elo
                direction = "+" if delta >= 0 else ""
                lines.append(f"  Elo: {first_elo:.0f} -> {last_elo:.0f} ({direction}{delta:.0f})")
            else:
                lines.append("  Elo: Not yet evaluated")

            # Show lineage
            lines.append(f"  Lineage: " + " -> ".join(f"v{g.version}" for g in gens))

            # Total training data
            total_games = sum(g.training_games or 0 for g in gens)
            total_samples = sum(g.training_samples or 0 for g in gens)
            lines.append(f"  Training: {total_games:,} games, {total_samples:,} samples")

            lines.append("")

        return "\n".join(lines)

    def get_next_version(self, board_type: str, num_players: int) -> int:
        """Get the next version number for a board configuration.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players

        Returns:
            Next version number (e.g., 1 if no versions exist, 4 if v3 is latest)
        """
        latest = self.get_latest_generation(board_type, num_players)
        if latest:
            return latest.version + 1
        return 1

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM model_generations")
            total_generations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM generation_tournaments")
            total_tournaments = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM elo_progression")
            total_elo_snapshots = cursor.fetchone()[0]

            cursor.execute("""
                SELECT board_type, num_players, COUNT(*) as count
                FROM model_generations
                GROUP BY board_type, num_players
            """)
            by_config = {f"{row[0]}_{row[1]}p": row[2] for row in cursor.fetchall()}

        return {
            "total_generations": total_generations,
            "total_tournaments": total_tournaments,
            "total_elo_snapshots": total_elo_snapshots,
            "generations_by_config": by_config,
        }


# Singleton accessor
_tracker_instance: GenerationTracker | None = None


def get_generation_tracker(db_path: Path | str | None = None) -> GenerationTracker:
    """Get the singleton GenerationTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = GenerationTracker(db_path)
    return _tracker_instance


def reset_generation_tracker() -> None:
    """Reset the singleton instance (for testing)."""
    global _tracker_instance
    _tracker_instance = None
