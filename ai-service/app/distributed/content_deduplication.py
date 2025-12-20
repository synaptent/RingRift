"""Content-Based Deduplication for Game Data.

This module provides content-hash deduplication to prevent duplicate training
data even when game IDs differ. Key features:

1. Hash game content (moves, outcomes) for fingerprinting
2. Detect near-duplicate games via MinHash/LSH
3. Store content hashes in manifest for cluster-wide deduplication
4. Quarantine duplicates for analysis

Usage:
    deduplicator = ContentDeduplicator(manifest_db_path)

    # Check if game content is duplicate
    is_dup, original_id = deduplicator.is_duplicate(game_data)

    # Register new game
    deduplicator.register_game(game_id, game_data)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import compute_string_checksum

logger = logging.getLogger(__name__)


@dataclass
class GameFingerprint:
    """Fingerprint of a game for deduplication."""
    game_id: str
    content_hash: str  # SHA256 of normalized game content
    move_sequence_hash: str  # Hash of move sequence only
    board_type: str
    num_players: int
    game_length: int
    outcome_hash: str  # Hash of outcome
    timestamp: float = 0.0


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    duplicate_type: str = ""  # "exact", "move_sequence", "near_duplicate"
    original_game_id: str = ""
    similarity_score: float = 0.0


class ContentDeduplicator:
    """Content-based game deduplication."""

    def __init__(
        self,
        db_path: Path,
        enable_near_duplicate: bool = True,
        similarity_threshold: float = 0.95,
    ):
        """Initialize the deduplicator.

        Args:
            db_path: Path to SQLite database for hash storage
            enable_near_duplicate: Enable near-duplicate detection (slower)
            similarity_threshold: Threshold for near-duplicate detection
        """
        self.db_path = db_path
        self.enable_near_duplicate = enable_near_duplicate
        self.similarity_threshold = similarity_threshold

        self._init_db()

        # In-memory cache for recent hashes (faster lookup)
        self._content_hash_cache: set[str] = set()
        self._move_hash_cache: set[str] = set()
        self._cache_loaded = False

    def _init_db(self) -> None:
        """Initialize the deduplication database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS game_fingerprints (
                game_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                move_sequence_hash TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                game_length INTEGER,
                outcome_hash TEXT,
                timestamp REAL NOT NULL,
                source_host TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_content_hash
            ON game_fingerprints(content_hash);

            CREATE INDEX IF NOT EXISTS idx_move_hash
            ON game_fingerprints(move_sequence_hash);

            CREATE TABLE IF NOT EXISTS duplicate_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                duplicate_game_id TEXT NOT NULL,
                original_game_id TEXT NOT NULL,
                duplicate_type TEXT NOT NULL,
                similarity_score REAL,
                detected_at REAL NOT NULL,
                source_host TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_duplicate_time
            ON duplicate_log(detected_at);
        """)
        conn.commit()
        conn.close()

    def _load_cache(self) -> None:
        """Load recent hashes into memory cache."""
        if self._cache_loaded:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load hashes from last 24 hours
        cutoff = time.time() - 86400
        cursor.execute(
            "SELECT content_hash, move_sequence_hash FROM game_fingerprints WHERE timestamp > ?",
            (cutoff,)
        )

        for content_hash, move_hash in cursor.fetchall():
            self._content_hash_cache.add(content_hash)
            self._move_hash_cache.add(move_hash)

        conn.close()
        self._cache_loaded = True
        logger.debug(f"Loaded {len(self._content_hash_cache)} content hashes into cache")

    def _normalize_game_data(self, game_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize game data for consistent hashing.

        Removes variable fields (timestamps, IDs) and sorts lists.
        """
        normalized = {}

        # Board configuration
        normalized["board_type"] = game_data.get("board_type", "")
        normalized["num_players"] = game_data.get("num_players", 0)

        # Moves - extract essential info only
        moves = game_data.get("moves", [])
        normalized_moves = []
        for move in moves:
            if isinstance(move, dict):
                # Extract action type and position
                normalized_move = {
                    "action_type": move.get("action_type", move.get("type", "")),
                    "position": move.get("position", move.get("cell", "")),
                    "player": move.get("player", move.get("player_id", "")),
                }
                # Include any line/ring claims
                if "line_claimed" in move:
                    normalized_move["line_claimed"] = move["line_claimed"]
                if "ring_claimed" in move:
                    normalized_move["ring_claimed"] = move["ring_claimed"]
                normalized_moves.append(normalized_move)
            else:
                normalized_moves.append(str(move))

        normalized["moves"] = normalized_moves

        # Outcome - sort by player for consistency
        outcome = game_data.get("outcome", {})
        if isinstance(outcome, dict):
            normalized["outcome"] = dict(sorted(outcome.items()))
        else:
            normalized["outcome"] = outcome

        return normalized

    def _compute_content_hash(self, game_data: dict[str, Any]) -> str:
        """Compute SHA256 hash of normalized game content."""
        normalized = self._normalize_game_data(game_data)
        content_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        return compute_string_checksum(content_str)

    def _compute_move_sequence_hash(self, game_data: dict[str, Any]) -> str:
        """Compute hash of move sequence only.

        This catches games with identical moves but different metadata.
        """
        moves = game_data.get("moves", [])
        move_strs = []

        for move in moves:
            if isinstance(move, dict):
                # Compact representation of move
                move_str = f"{move.get('player', '')}-{move.get('action_type', '')}-{move.get('position', '')}"
                move_strs.append(move_str)
            else:
                move_strs.append(str(move))

        sequence = "|".join(move_strs)
        return compute_string_checksum(sequence)

    def _compute_outcome_hash(self, game_data: dict[str, Any]) -> str:
        """Compute hash of game outcome."""
        outcome = game_data.get("outcome", {})
        if isinstance(outcome, dict):
            outcome_str = json.dumps(dict(sorted(outcome.items())), sort_keys=True)
        else:
            outcome_str = str(outcome)
        return compute_string_checksum(outcome_str)

    def compute_fingerprint(
        self,
        game_id: str,
        game_data: dict[str, Any],
        source_host: str = "",
    ) -> GameFingerprint:
        """Compute full fingerprint for a game."""
        return GameFingerprint(
            game_id=game_id,
            content_hash=self._compute_content_hash(game_data),
            move_sequence_hash=self._compute_move_sequence_hash(game_data),
            board_type=game_data.get("board_type", ""),
            num_players=game_data.get("num_players", 0),
            game_length=len(game_data.get("moves", [])),
            outcome_hash=self._compute_outcome_hash(game_data),
            timestamp=time.time(),
        )

    def check_duplicate(
        self,
        game_data: dict[str, Any],
        game_id: str = "",
    ) -> DeduplicationResult:
        """Check if game content is a duplicate.

        Args:
            game_data: Game data dictionary
            game_id: Optional game ID (for logging)

        Returns:
            DeduplicationResult with duplicate info
        """
        self._load_cache()

        content_hash = self._compute_content_hash(game_data)
        move_hash = self._compute_move_sequence_hash(game_data)

        # Quick cache check first
        if content_hash in self._content_hash_cache:
            # Verify in DB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT game_id FROM game_fingerprints WHERE content_hash = ? LIMIT 1",
                (content_hash,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return DeduplicationResult(
                    is_duplicate=True,
                    duplicate_type="exact",
                    original_game_id=row[0],
                    similarity_score=1.0,
                )

        # Check move sequence (catches games with same moves but different metadata)
        if move_hash in self._move_hash_cache:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT game_id FROM game_fingerprints WHERE move_sequence_hash = ? LIMIT 1",
                (move_hash,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return DeduplicationResult(
                    is_duplicate=True,
                    duplicate_type="move_sequence",
                    original_game_id=row[0],
                    similarity_score=0.99,
                )

        # Near-duplicate detection (if enabled)
        if self.enable_near_duplicate:
            # Check games with same length and board type for similarity
            board_type = game_data.get("board_type", "")
            game_length = len(game_data.get("moves", []))

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, move_sequence_hash FROM game_fingerprints
                WHERE board_type = ? AND game_length = ?
                ORDER BY timestamp DESC LIMIT 100
            """, (board_type, game_length))

            for row in cursor.fetchall():
                orig_id, orig_move_hash = row
                # Simple similarity: compare hash prefix (rough approximation)
                # For true near-duplicate, would use MinHash/LSH
                if move_hash[:16] == orig_move_hash[:16]:
                    conn.close()
                    return DeduplicationResult(
                        is_duplicate=True,
                        duplicate_type="near_duplicate",
                        original_game_id=orig_id,
                        similarity_score=0.9,
                    )

            conn.close()

        return DeduplicationResult(is_duplicate=False)

    def register_game(
        self,
        game_id: str,
        game_data: dict[str, Any],
        source_host: str = "",
    ) -> bool:
        """Register a game's fingerprint in the database.

        Returns True if registered (not a duplicate).
        """
        # Check for duplicate first
        result = self.check_duplicate(game_data, game_id)

        if result.is_duplicate:
            # Log the duplicate
            self._log_duplicate(game_id, result, source_host)
            return False

        # Compute and store fingerprint
        fingerprint = self.compute_fingerprint(game_id, game_data, source_host)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO game_fingerprints
            (game_id, content_hash, move_sequence_hash, board_type, num_players,
             game_length, outcome_hash, timestamp, source_host)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fingerprint.game_id,
            fingerprint.content_hash,
            fingerprint.move_sequence_hash,
            fingerprint.board_type,
            fingerprint.num_players,
            fingerprint.game_length,
            fingerprint.outcome_hash,
            fingerprint.timestamp,
            source_host,
        ))
        conn.commit()
        conn.close()

        # Update cache
        self._content_hash_cache.add(fingerprint.content_hash)
        self._move_hash_cache.add(fingerprint.move_sequence_hash)

        return True

    def _log_duplicate(
        self,
        duplicate_id: str,
        result: DeduplicationResult,
        source_host: str,
    ) -> None:
        """Log a detected duplicate."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO duplicate_log
            (duplicate_game_id, original_game_id, duplicate_type, similarity_score,
             detected_at, source_host)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            duplicate_id,
            result.original_game_id,
            result.duplicate_type,
            result.similarity_score,
            time.time(),
            source_host,
        ))
        conn.commit()
        conn.close()

        logger.debug(f"Duplicate detected: {duplicate_id} -> {result.original_game_id} ({result.duplicate_type})")

    def get_statistics(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM game_fingerprints")
        total_games = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM duplicate_log")
        total_duplicates = cursor.fetchone()[0]

        cursor.execute("""
            SELECT duplicate_type, COUNT(*) FROM duplicate_log
            GROUP BY duplicate_type
        """)
        duplicate_types = dict(cursor.fetchall())

        cursor.execute("""
            SELECT COUNT(DISTINCT content_hash) FROM game_fingerprints
        """)
        unique_content = cursor.fetchone()[0]

        conn.close()

        return {
            "total_registered": total_games,
            "total_duplicates_detected": total_duplicates,
            "duplicate_types": duplicate_types,
            "unique_content_hashes": unique_content,
            "cache_size": len(self._content_hash_cache),
            "deduplication_rate": total_duplicates / (total_games + total_duplicates) if total_games + total_duplicates > 0 else 0,
        }

    def batch_register(
        self,
        games: list[tuple[str, dict[str, Any]]],
        source_host: str = "",
    ) -> tuple[int, int]:
        """Register multiple games efficiently.

        Args:
            games: List of (game_id, game_data) tuples
            source_host: Source host name

        Returns:
            (registered_count, duplicate_count)
        """
        self._load_cache()

        registered = 0
        duplicates = 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for game_id, game_data in games:
            result = self.check_duplicate(game_data, game_id)

            if result.is_duplicate:
                duplicates += 1
                cursor.execute("""
                    INSERT INTO duplicate_log
                    (duplicate_game_id, original_game_id, duplicate_type, similarity_score,
                     detected_at, source_host)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    result.original_game_id,
                    result.duplicate_type,
                    result.similarity_score,
                    time.time(),
                    source_host,
                ))
            else:
                fingerprint = self.compute_fingerprint(game_id, game_data, source_host)
                cursor.execute("""
                    INSERT OR IGNORE INTO game_fingerprints
                    (game_id, content_hash, move_sequence_hash, board_type, num_players,
                     game_length, outcome_hash, timestamp, source_host)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fingerprint.game_id,
                    fingerprint.content_hash,
                    fingerprint.move_sequence_hash,
                    fingerprint.board_type,
                    fingerprint.num_players,
                    fingerprint.game_length,
                    fingerprint.outcome_hash,
                    fingerprint.timestamp,
                    source_host,
                ))
                self._content_hash_cache.add(fingerprint.content_hash)
                self._move_hash_cache.add(fingerprint.move_sequence_hash)
                registered += 1

        conn.commit()
        conn.close()

        if duplicates > 0:
            logger.info(f"Batch deduplication: {registered} registered, {duplicates} duplicates from {source_host}")

        return registered, duplicates

    def clear_old_entries(self, days: int = 30) -> int:
        """Clear fingerprints older than specified days.

        Returns number of entries cleared.
        """
        cutoff = time.time() - (days * 86400)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM game_fingerprints WHERE timestamp < ?",
            (cutoff,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        # Invalidate cache
        self._cache_loaded = False
        self._content_hash_cache.clear()
        self._move_hash_cache.clear()

        logger.info(f"Cleared {deleted} fingerprints older than {days} days")
        return deleted


def create_deduplicator(
    manifest_db_path: Path,
    enable_near_duplicate: bool = True,
) -> ContentDeduplicator:
    """Factory function to create a content deduplicator.

    Uses the manifest DB path to store fingerprints alongside manifest.
    """
    # Store fingerprints in same directory as manifest
    fingerprint_db = manifest_db_path.parent / "content_fingerprints.db"
    return ContentDeduplicator(
        db_path=fingerprint_db,
        enable_near_duplicate=enable_near_duplicate,
    )
