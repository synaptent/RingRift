"""Incremental NPZ export system for efficient training data generation.

.. deprecated::
    This module is deprecated since December 2025.
    Use `app.training.export_cache.ExportCache` instead for simpler
    file-level caching. This module will be removed in Q2 2026.

    Migration:
        # Old (incremental_export):
        exporter = IncrementalExporter("square8", 2)
        new_games = exporter.get_unexported_game_ids(all_game_ids)

        # New (export_cache):
        cache = ExportCache()
        if cache.needs_export(db_paths, output_path, board_type, num_players):
            # Perform export
            pass

This module provides incremental export functionality that:
1. Tracks which game_ids have already been exported (SQLite-backed)
2. Only processes new games since last export
3. Uses sharded NPZ files for efficient appending
4. Supports merging shards into consolidated training files

This dramatically reduces export time from minutes to seconds by avoiding
re-processing of already-exported games.

Usage:
    from app.training.incremental_export import IncrementalExporter

    exporter = IncrementalExporter("square8", 2)
    new_games = exporter.get_unexported_game_ids(all_game_ids)
    # ... process only new_games ...
    exporter.mark_exported(new_game_ids)
"""
import warnings

warnings.warn(
    "app.training.incremental_export is deprecated since December 2025. "
    "Use app.training.export_cache.ExportCache instead for simpler file-level caching. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TRACKER_DIR = "data/training/export_state"
DEFAULT_SHARD_DIR = "data/training/shards"


@dataclass
class ExportStats:
    """Statistics from an incremental export run."""
    total_games_seen: int = 0
    new_games_found: int = 0
    games_exported: int = 0
    positions_extracted: int = 0
    shards_created: int = 0
    export_time_seconds: float = 0.0


class GameIdTracker:
    """SQLite-backed tracker for exported game IDs.

    Efficiently tracks which games have been exported to avoid
    re-processing the same games on subsequent export runs.
    """

    def __init__(self, board_type: str, num_players: int, tracker_dir: str = DEFAULT_TRACKER_DIR):
        """Initialize the game ID tracker.

        Args:
            board_type: Board type (square8, hex8, etc.)
            num_players: Number of players
            tracker_dir: Directory for tracker database
        """
        self.board_type = board_type
        self.num_players = num_players
        self.config_key = f"{board_type}_{num_players}p"

        # Ensure tracker directory exists
        self.tracker_dir = Path(tracker_dir)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)

        # Database path
        self.db_path = self.tracker_dir / f"exported_games_{self.config_key}.db"

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exported_games (
                game_id TEXT PRIMARY KEY,
                source_file TEXT,
                export_timestamp REAL,
                shard_file TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_export_timestamp
            ON exported_games(export_timestamp)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS export_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                games_exported INTEGER,
                positions_extracted INTEGER,
                shard_file TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_exported_game_ids(self) -> set[str]:
        """Get all previously exported game IDs."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT game_id FROM exported_games")
        game_ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return game_ids

    def get_unexported_game_ids(self, all_game_ids: set[str]) -> set[str]:
        """Get game IDs that haven't been exported yet.

        Args:
            all_game_ids: Set of all available game IDs

        Returns:
            Set of game IDs that need to be exported
        """
        exported = self.get_exported_game_ids()
        return all_game_ids - exported

    def mark_exported(
        self,
        game_ids: list[str],
        source_file: str = "",
        shard_file: str = ""
    ):
        """Mark games as exported.

        Args:
            game_ids: List of game IDs that were exported
            source_file: Source file the games came from
            shard_file: Shard file they were exported to
        """
        if not game_ids:
            return

        timestamp = time.time()
        conn = sqlite3.connect(str(self.db_path))

        # Insert in batches for efficiency
        batch_size = 1000
        for i in range(0, len(game_ids), batch_size):
            batch = game_ids[i:i + batch_size]
            conn.executemany(
                """INSERT OR REPLACE INTO exported_games
                   (game_id, source_file, export_timestamp, shard_file)
                   VALUES (?, ?, ?, ?)""",
                [(gid, source_file, timestamp, shard_file) for gid in batch]
            )

        conn.commit()
        conn.close()

        logger.info(f"Marked {len(game_ids)} games as exported for {self.config_key}")

    def record_export_run(
        self,
        games_exported: int,
        positions_extracted: int,
        shard_file: str
    ):
        """Record an export run for tracking."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO export_runs
               (timestamp, games_exported, positions_extracted, shard_file)
               VALUES (?, ?, ?, ?)""",
            (time.time(), games_exported, positions_extracted, shard_file)
        )
        conn.commit()
        conn.close()

    def get_export_count(self) -> int:
        """Get total number of exported games."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM exported_games")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def clear(self):
        """Clear all export tracking data (use with caution)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM exported_games")
        conn.execute("DELETE FROM export_runs")
        conn.commit()
        conn.close()
        logger.warning(f"Cleared export tracking for {self.config_key}")


class ShardManager:
    """Manages NPZ shards for incremental training data.

    Instead of regenerating full NPZ files, this creates small shards
    that can be efficiently merged when needed.
    """

    def __init__(
        self,
        board_type: str,
        num_players: int,
        shard_dir: str = DEFAULT_SHARD_DIR,
        max_samples_per_shard: int = 50000,
    ):
        """Initialize the shard manager.

        Args:
            board_type: Board type
            num_players: Number of players
            shard_dir: Directory for shard files
            max_samples_per_shard: Maximum samples per shard file
        """
        self.board_type = board_type
        self.num_players = num_players
        self.config_key = f"{board_type}_{num_players}p"
        self.max_samples_per_shard = max_samples_per_shard

        # Shard directory
        self.shard_dir = Path(shard_dir) / self.config_key
        self.shard_dir.mkdir(parents=True, exist_ok=True)

    def get_next_shard_path(self) -> Path:
        """Get path for the next shard file."""
        existing = list(self.shard_dir.glob("shard_*.npz"))
        next_num = len(existing)
        return self.shard_dir / f"shard_{next_num:05d}.npz"

    def save_shard(
        self,
        features: np.ndarray,
        globals_vec: np.ndarray,
        values: np.ndarray,
        values_mp: np.ndarray,
        num_players: np.ndarray,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        move_numbers: np.ndarray,
        total_game_moves: np.ndarray,
        phases: np.ndarray,
    ) -> Path:
        """Save a shard of training data.

        Returns:
            Path to the saved shard file
        """
        shard_path = self.get_next_shard_path()

        np.savez_compressed(
            shard_path,
            features=features,
            globals=globals_vec,
            values=values,
            values_mp=values_mp,
            num_players=num_players,
            policy_indices=policy_indices,
            policy_values=policy_values,
            move_numbers=move_numbers,
            total_game_moves=total_game_moves,
            phases=phases,
        )

        logger.info(f"Saved shard with {len(features)} samples to {shard_path}")
        return shard_path

    def get_all_shards(self) -> list[Path]:
        """Get all shard files in order."""
        return sorted(self.shard_dir.glob("shard_*.npz"))

    def merge_shards(self, output_path: Path, max_samples: int | None = None) -> int:
        """Merge all shards into a single NPZ file.

        Args:
            output_path: Output file path
            max_samples: Maximum samples to include (None = all)

        Returns:
            Number of samples in merged file
        """
        shards = self.get_all_shards()
        if not shards:
            logger.warning("No shards to merge")
            return 0

        logger.info(f"Merging {len(shards)} shards...")

        # Collect all data
        all_features = []
        all_globals = []
        all_values = []
        all_values_mp = []
        all_num_players = []
        all_policy_indices = []
        all_policy_values = []
        all_move_numbers = []
        all_total_game_moves = []
        all_phases = []

        total_samples = 0
        for shard_path in shards:
            if max_samples and total_samples >= max_samples:
                break

            with np.load(shard_path, allow_pickle=True) as data:
                n_samples = len(data["features"])

                # Limit samples if needed
                if max_samples:
                    n_to_take = min(n_samples, max_samples - total_samples)
                else:
                    n_to_take = n_samples

                all_features.append(data["features"][:n_to_take])
                all_globals.append(data["globals"][:n_to_take])
                all_values.append(data["values"][:n_to_take])
                all_values_mp.append(data["values_mp"][:n_to_take])
                all_num_players.append(data["num_players"][:n_to_take])
                all_policy_indices.extend(data["policy_indices"][:n_to_take])
                all_policy_values.extend(data["policy_values"][:n_to_take])
                all_move_numbers.append(data["move_numbers"][:n_to_take])
                all_total_game_moves.append(data["total_game_moves"][:n_to_take])
                all_phases.extend(data["phases"][:n_to_take])

                total_samples += n_to_take

        if total_samples == 0:
            logger.warning("No samples found in shards")
            return 0

        # Concatenate arrays
        features_arr = np.concatenate(all_features, axis=0)
        globals_arr = np.concatenate(all_globals, axis=0)
        values_arr = np.concatenate(all_values, axis=0)
        values_mp_arr = np.concatenate(all_values_mp, axis=0)
        num_players_arr = np.concatenate(all_num_players, axis=0)
        move_numbers_arr = np.concatenate(all_move_numbers, axis=0)
        total_game_moves_arr = np.concatenate(all_total_game_moves, axis=0)
        policy_indices_arr = np.array(all_policy_indices, dtype=object)
        policy_values_arr = np.array(all_policy_values, dtype=object)
        phases_arr = np.array(all_phases, dtype=object)

        # Save merged file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            features=features_arr,
            globals=globals_arr,
            values=values_arr,
            policy_indices=policy_indices_arr,
            policy_values=policy_values_arr,
            move_numbers=move_numbers_arr,
            total_game_moves=total_game_moves_arr,
            phases=phases_arr,
            values_mp=values_mp_arr,
            num_players=num_players_arr,
        )

        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Merged {total_samples} samples into {output_path} ({output_size_mb:.2f} MB)")

        return total_samples

    def get_total_samples(self) -> int:
        """Get total samples across all shards."""
        total = 0
        for shard_path in self.get_all_shards():
            try:
                with np.load(shard_path, allow_pickle=True) as data:
                    total += len(data["features"])
            except Exception:
                pass
        return total

    def cleanup_old_shards(self, keep_last_n: int = 10):
        """Remove old shards, keeping only the most recent ones."""
        shards = self.get_all_shards()
        if len(shards) <= keep_last_n:
            return

        to_remove = shards[:-keep_last_n]
        for shard_path in to_remove:
            try:
                shard_path.unlink()
                logger.info(f"Removed old shard: {shard_path}")
            except Exception as e:
                logger.warning(f"Failed to remove shard {shard_path}: {e}")


class IncrementalExporter:
    """High-level interface for incremental NPZ export.

    Combines game ID tracking and shard management for efficient
    incremental training data generation.
    """

    def __init__(
        self,
        board_type: str,
        num_players: int,
        tracker_dir: str = DEFAULT_TRACKER_DIR,
        shard_dir: str = DEFAULT_SHARD_DIR,
        max_samples_per_shard: int = 50000,
    ):
        """Initialize the incremental exporter.

        Args:
            board_type: Board type
            num_players: Number of players
            tracker_dir: Directory for export tracking database
            shard_dir: Directory for NPZ shards
            max_samples_per_shard: Maximum samples per shard
        """
        self.board_type = board_type
        self.num_players = num_players
        self.config_key = f"{board_type}_{num_players}p"

        self.tracker = GameIdTracker(board_type, num_players, tracker_dir)
        self.shard_manager = ShardManager(
            board_type, num_players, shard_dir, max_samples_per_shard
        )

    def get_unexported_game_ids(self, all_game_ids: set[str]) -> set[str]:
        """Get game IDs that haven't been exported yet."""
        return self.tracker.get_unexported_game_ids(all_game_ids)

    def mark_exported(self, game_ids: list[str], source_file: str = "", shard_file: str = ""):
        """Mark games as exported."""
        self.tracker.mark_exported(game_ids, source_file, shard_file)

    def save_shard(
        self,
        features: np.ndarray,
        globals_vec: np.ndarray,
        values: np.ndarray,
        values_mp: np.ndarray,
        num_players: np.ndarray,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        move_numbers: np.ndarray,
        total_game_moves: np.ndarray,
        phases: np.ndarray,
        game_ids: list[str],
        source_file: str = "",
    ) -> Path:
        """Save training data as a shard and mark games as exported.

        Args:
            features: Feature arrays
            globals_vec: Global feature arrays
            values: Value targets
            values_mp: Multi-player value targets
            num_players: Player count per sample
            policy_indices: Policy indices
            policy_values: Policy probabilities
            move_numbers: Move numbers
            total_game_moves: Total game moves
            phases: Game phases
            game_ids: Game IDs in this shard
            source_file: Source file name

        Returns:
            Path to saved shard
        """
        shard_path = self.shard_manager.save_shard(
            features, globals_vec, values, values_mp, num_players,
            policy_indices, policy_values, move_numbers, total_game_moves, phases
        )

        self.tracker.mark_exported(game_ids, source_file, str(shard_path))
        self.tracker.record_export_run(len(game_ids), len(features), str(shard_path))

        return shard_path

    def merge_to_npz(self, output_path: Path, max_samples: int | None = None) -> int:
        """Merge all shards into a single NPZ file."""
        return self.shard_manager.merge_shards(output_path, max_samples)

    def get_stats(self) -> dict[str, Any]:
        """Get export statistics."""
        return {
            "config": self.config_key,
            "total_exported_games": self.tracker.get_export_count(),
            "total_samples": self.shard_manager.get_total_samples(),
            "num_shards": len(self.shard_manager.get_all_shards()),
        }


def get_incremental_exporter(board_type: str, num_players: int) -> IncrementalExporter:
    """Factory function to get an incremental exporter for a config."""
    return IncrementalExporter(board_type, num_players)
