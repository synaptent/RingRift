"""Unified Game Discovery - Find all game databases across cluster paths.

This module provides a centralized way to discover game databases regardless
of where they're stored. It handles all known storage patterns:

1. Central databases: data/games/selfplay.db, data/games/jsonl_aggregated.db
2. Per-config databases: data/games/{board_type}_{num_players}.db
3. Tournament databases: data/games/tournament_{board_type}_{num_players}.db
4. Gauntlet databases: data/games/gauntlet_{board_type}_{num_players}p.db
5. Canonical databases: data/selfplay/canonical_{board_type}_{num_players}.db
6. Unified selfplay: data/selfplay/unified_*/games.db
7. P2P selfplay: data/selfplay/p2p/{board_type}_{num_players}*/*/games.db
8. P2P hybrid: data/selfplay/p2p_hybrid/{board_type}_{num_players}/*/games.db
9. P2P GPU selfplay: data/selfplay/p2p_gpu/{board_type}_{num_players}p*/*/games.db (Jan 2026)
10. Gumbel selfplay: data/selfplay/gumbel/{board_type}_{num_players}p/*/games.db (Jan 2026)
11. SLURM HPC: data/selfplay/slurm_{board_type}_{num_players}p*/games.db (Jan 2026)
12. Harvested data: data/training/*/harvested_games.db
13. OWC imports: data/games/owc_imports/*.db (Dec 30, 2025)
14. Synced data: data/games/synced/**/*.db (Dec 30, 2025)

Usage:
    from app.utils.game_discovery import GameDiscovery

    # Find all databases
    discovery = GameDiscovery()
    all_dbs = discovery.find_all_databases()

    # Find databases for specific config
    hex8_2p_dbs = discovery.find_databases_for_config("hex8", 2)

    # Count games by config
    counts = discovery.count_games_by_config()
    print(counts)
    # {'square8_2p': 50000, 'hex8_2p': 34000, ...}

    # Get total games for a config
    total = discovery.get_total_games("hexagonal", 2)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from app.utils.parallel_defaults import maybe_parallel_map

logger = logging.getLogger(__name__)

# Board type aliases - some databases use different names
BOARD_TYPE_ALIASES = {
    "hex8": ["hex8"],
    "hexagonal": ["hexagonal", "hex"],
    "square8": ["square8", "sq8"],
    "square19": ["square19", "sq19"],
}

# All board types in canonical form
ALL_BOARD_TYPES = ["square8", "square19", "hexagonal", "hex8"]

# All player counts
ALL_PLAYER_COUNTS = [2, 3, 4]


@dataclass
class DatabaseInfo:
    """Information about a discovered game database."""

    path: Path
    board_type: str | None = None
    num_players: int | None = None
    game_count: int = 0
    is_central: bool = False  # Central DBs contain multiple board types
    source_pattern: str = ""  # Which pattern matched this DB
    config_counts: dict[tuple[str, int], int] = field(default_factory=dict)  # For central DBs


@dataclass
class JsonlFileInfo:
    """Information about a discovered JSONL game file (December 2025).

    JSONL files contain game records from GPU selfplay in JSON lines format.
    These are converted to NPZ via scripts/jsonl_to_npz.py.
    """

    path: Path
    board_type: str | None = None
    num_players: int | None = None
    game_count: int = 0
    source_pattern: str = ""

    def __post_init__(self):
        """Count games in JSONL file if not already set."""
        if self.game_count == 0 and self.path.exists():
            try:
                with open(self.path) as f:
                    self.game_count = sum(1 for _ in f)
            except (OSError, UnicodeDecodeError):
                # File read errors (permission, not found, encoding)
                pass


@dataclass
class GameCounts:
    """Game counts aggregated by configuration."""

    by_config: dict[str, int] = field(default_factory=dict)
    by_board_type: dict[str, int] = field(default_factory=dict)
    by_num_players: dict[int, int] = field(default_factory=dict)
    total: int = 0
    databases_found: int = 0


class GameDiscovery:
    """Unified game database discovery across all storage patterns."""

    # Database path patterns (relative to ai-service root)
    # Order matters - more specific patterns first
    DB_PATTERNS = [
        # Central databases (contain all board types)
        ("data/games/selfplay.db", True),
        ("data/games/jsonl_aggregated.db", True),
        # Per-config databases
        ("data/games/{board_type}_{num_players}p.db", False),
        ("data/games/{board_type}_{num_players}.db", False),
        # Tournament databases
        ("data/games/tournament_{board_type}_{num_players}p.db", False),
        ("data/games/tournament_{board_type}_{num_players}.db", False),
        # Gauntlet evaluation databases (Dec 29, 2025)
        ("data/games/gauntlet_{board_type}_{num_players}p.db", False),
        ("data/games/baseline_calibration_{board_type}_{num_players}p.db", False),
        # Canonical selfplay databases
        ("data/selfplay/canonical_{board_type}_{num_players}p.db", False),
        ("data/selfplay/canonical_{board_type}.db", True),  # May contain multiple player counts
        # Unified selfplay (session-based)
        ("data/selfplay/unified_*/games.db", True),
        # P2P selfplay
        ("data/selfplay/p2p/{board_type}_{num_players}p*/*/games.db", False),
        ("data/selfplay/p2p/{board_type}_{num_players}*/*/games.db", False),
        # P2P hybrid selfplay
        ("data/selfplay/p2p_hybrid/{board_type}_{num_players}p/*/games.db", False),
        ("data/selfplay/p2p_hybrid/{board_type}_{num_players}/*/games.db", False),
        # P2P GPU selfplay (critical - often has 10-50K games per config)
        ("data/selfplay/p2p_gpu/{board_type}_{num_players}p/*/games.db", False),
        ("data/selfplay/p2p_gpu/{board_type}_{num_players}p*/*/games.db", False),
        ("data/selfplay/p2p_gpu/*/{board_type}_{num_players}p/games.db", False),
        ("data/selfplay/p2p_gpu/*/{board_type}_{num_players}p*/games.db", False),
        # Gumbel MCTS selfplay
        ("data/selfplay/gumbel/{board_type}_{num_players}p/*/games.db", False),
        ("data/selfplay/gumbel/{board_type}_{num_players}/*/games.db", False),
        ("data/selfplay/gumbel/*/{board_type}_{num_players}p/games.db", False),
        # SLURM HPC output
        ("data/selfplay/slurm_{board_type}_{num_players}p*/games.db", False),
        ("data/selfplay/slurm_*/{board_type}_{num_players}p/games.db", False),
        ("data/selfplay/slurm_*/{board_type}_{num_players}p*/games.db", False),
        # Catch-all patterns for any board/player subdirs (fallback)
        ("data/selfplay/*/{board_type}_{num_players}p/*/games.db", False),
        ("data/selfplay/*/*/{board_type}_{num_players}p/games.db", False),
        ("data/selfplay/*/*/{board_type}_{num_players}p*/games.db", False),
        # Harvested training data
        ("data/training/*/harvested_games.db", True),
        # OWC imports (Dec 30, 2025) - Data imported from OWC external drive
        ("data/games/owc_imports/*.db", True),
        # Synced data from other cluster nodes (Dec 30, 2025)
        ("data/games/synced/*.db", True),
        ("data/games/synced/**/*.db", True),
        # Remote-fetched databases (Jan 2026) - from RemoteGameFetcher
        ("data/games/fetched/*.db", True),
        ("data/games/fetched/**/*.db", True),
        # S3 downloads (Jan 2026) - from aws s3 cp
        ("data/games/s3_downloads/*.db", True),
        ("data/games/s3_downloads/**/*.db", True),
        # Inter-node cluster sync (Jan 2026)
        ("data/games/cluster_sync/*.db", True),
        ("data/games/cluster_sync/**/*.db", True),
        # Synced databases in root data/games directory (Jan 2026)
        # Files like gh200_square19_3p_synced.db, nebius_hex8_2p_synced.db
        ("data/games/*_synced.db", True),
        # Legacy patterns
        ("data/games/hex8_*.db", False),
        ("data/games/canonical_*.db", True),
    ]

    # JSONL file patterns (December 2025) - for GPU selfplay data
    JSONL_PATTERNS = [
        # Cluster JSONL - GPU selfplay
        ("data/selfplay/cluster_jsonl/gpu/games_{board_type}_{num_players}p_*.jsonl", False),
        # P2P GPU selfplay
        ("data/selfplay/cluster_jsonl/p2p_gpu/{board_type}_{num_players}p/*/games_*.jsonl", False),
        ("data/selfplay/cluster_jsonl/p2p_gpu/{board_type}_{num_players}/*/games_*.jsonl", False),
        # Hex variants
        ("data/selfplay/cluster_jsonl/p2p_gpu/hex_{num_players}p/*/games_hexagonal_*.jsonl", False),
        # Logs/selfplay (soak tests)
        ("logs/selfplay/soak.*.{board_type}.{num_players}p.*.jsonl", False),
        # Cluster JSONL directory - catch-all
        ("cluster_jsonl/*.jsonl", False),
        # Legacy GPU training
        ("data/gpu_training_*.jsonl", False),
    ]

    def __init__(self, root_path: Path | str | None = None):
        """Initialize game discovery.

        Args:
            root_path: Root path to ai-service directory. If None, auto-detect.
        """
        if root_path is None:
            # Auto-detect based on common locations
            candidates = [
                Path(__file__).parent.parent.parent,  # From app/utils/
                Path.cwd(),
                Path.home() / "ringrift" / "ai-service",
                Path("/workspace/ringrift/ai-service"),
                Path("/lambda/nfs/RingRift/ai-service"),
            ]
            for candidate in candidates:
                if (candidate / "data").exists():
                    root_path = candidate
                    break
            else:
                root_path = Path.cwd()

        self.root_path = Path(root_path)
        self._cache: dict[str, list[DatabaseInfo]] = {}

    def find_all_databases(self, use_cache: bool = True) -> list[DatabaseInfo]:
        """Find all game databases using known patterns.

        Args:
            use_cache: If True, use cached results if available.

        Returns:
            List of DatabaseInfo objects for all found databases.
        """
        cache_key = "all"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        databases = []
        seen_paths: set[Path] = set()

        for pattern, is_central in self.DB_PATTERNS:
            for db_info in self._find_by_pattern(pattern, is_central):
                if db_info.path not in seen_paths:
                    seen_paths.add(db_info.path)
                    databases.append(db_info)

        self._cache[cache_key] = databases
        return databases

    def find_databases_for_config(
        self,
        board_type: str,
        num_players: int,
        include_central: bool = True,
        use_cache: bool = True,
    ) -> list[DatabaseInfo]:
        """Find databases containing games for a specific configuration.

        Args:
            board_type: Board type (square8, square19, hexagonal, hex8)
            num_players: Number of players (2, 3, 4)
            include_central: Include central databases that contain multiple configs
            use_cache: Use cached results if available

        Returns:
            List of DatabaseInfo objects matching the configuration.
        """
        cache_key = f"{board_type}_{num_players}p_central={include_central}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        all_dbs = self.find_all_databases(use_cache=use_cache)
        matching = []

        # Get aliases for this board type
        aliases = BOARD_TYPE_ALIASES.get(board_type, [board_type])

        for db_info in all_dbs:
            # Central databases need to be queried
            if db_info.is_central:
                if include_central:
                    # Check if this DB actually has games for this config
                    count = self._count_games_in_db(
                        db_info.path, board_type, num_players
                    )
                    if count > 0:
                        db_copy = DatabaseInfo(
                            path=db_info.path,
                            board_type=board_type,
                            num_players=num_players,
                            game_count=count,
                            is_central=True,
                            source_pattern=db_info.source_pattern,
                        )
                        matching.append(db_copy)
            else:
                # Check if DB matches this config
                if db_info.board_type in aliases and db_info.num_players == num_players:
                    matching.append(db_info)

        self._cache[cache_key] = matching
        return matching

    def _count_single_db(self, db_info: DatabaseInfo) -> dict[str, int]:
        """Count games in a single database by config (helper for parallel execution).

        Jan 12, 2026: Extracted for parallel execution in count_games_by_config().

        Args:
            db_info: Database info object

        Returns:
            Dict mapping config_key (e.g., "hex8_2p") to game count
        """
        if not db_info.path.exists():
            return {}

        try:
            return self._get_config_counts(db_info.path)
        except (sqlite3.Error, OSError, PermissionError) as e:
            logger.debug(f"Error querying {db_info.path}: {e}")
            return {}

    def count_games_by_config(self, use_cache: bool = True) -> GameCounts:
        """Count games for all board/player configurations.

        Jan 12, 2026: Now uses parallel execution for database queries.
        Use RINGRIFT_FORCE_SEQUENTIAL=true to disable for debugging.

        Returns:
            GameCounts with breakdown by config, board type, and player count.
        """
        counts = GameCounts()
        all_dbs = self.find_all_databases(use_cache=use_cache)
        counts.databases_found = len(all_dbs)

        # Query databases in parallel (I/O-bound work)
        # maybe_parallel_map uses threads by default, falls back to sequential
        # if RINGRIFT_FORCE_SEQUENTIAL=true or < 4 databases
        all_config_counts = maybe_parallel_map(
            self._count_single_db,
            all_dbs,
            parallel_threshold=4,
            use_processes=False,  # Threads for I/O-bound SQLite queries
        )

        # Aggregate results from all databases
        for config_counts in all_config_counts:
            for config_key, count in config_counts.items():
                if count > 0:
                    counts.by_config[config_key] = (
                        counts.by_config.get(config_key, 0) + count
                    )
                    counts.total += count

                    # Parse config key
                    parts = config_key.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        try:
                            num_players = int(parts[1].replace("p", ""))
                            counts.by_board_type[board_type] = (
                                counts.by_board_type.get(board_type, 0) + count
                            )
                            counts.by_num_players[num_players] = (
                                counts.by_num_players.get(num_players, 0) + count
                            )
                        except ValueError:
                            pass

        return counts

    def get_total_games(
        self, board_type: str, num_players: int, use_cache: bool = True
    ) -> int:
        """Get total game count for a specific configuration.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            Total number of completed games for this configuration.
        """
        databases = self.find_databases_for_config(
            board_type, num_players, include_central=True, use_cache=use_cache
        )
        return sum(db.game_count for db in databases)

    def get_unified_total(
        self, board_type: str, num_players: int, include_external: bool = True
    ) -> int:
        """Get total games across ALL sources (local + cluster + OWC + S3).

        Jan 2026: Added for unified cluster data visibility.

        This method queries the UnifiedDataRegistry which aggregates:
        - Local game databases (this node)
        - Cluster-wide data (from P2P manifest broadcast)
        - OWC external drive (mac-studio)
        - AWS S3 bucket (archived data)

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)
            include_external: Include OWC and S3 sources (default: True)

        Returns:
            Total games across all sources. Falls back to local-only if
            registry unavailable.
        """
        try:
            from app.distributed.data_catalog import get_data_registry

            registry = get_data_registry()
            total = registry.get_total_games(board_type, num_players)

            # If registry has data, return it
            if total > 0:
                return total

            # Fall back to local-only if registry is empty
            # (e.g., P2P not running, no manifest received)
            return self.get_total_games(board_type, num_players)

        except ImportError:
            # Registry not available, fall back to local
            return self.get_total_games(board_type, num_players)
        except Exception as e:
            logger.debug(f"Unified registry query failed, falling back to local: {e}")
            return self.get_total_games(board_type, num_players)

    def clear_cache(self):
        """Clear the discovery cache."""
        self._cache.clear()

    # =========================================================================
    # Event-Driven Cache Invalidation (Dec 30, 2025)
    # Phase 1.3 of Distributed Data Pipeline Architecture
    # =========================================================================

    _event_wired: bool = False

    def wire_to_events(self) -> None:
        """Subscribe to data change events for automatic cache invalidation.

        Dec 30, 2025: Added for Phase 1.3 of distributed data pipeline.

        Subscribes to:
        - DATA_SYNC_COMPLETED: New data synced from other nodes
        - SELFPLAY_COMPLETE: New games generated locally
        - NPZ_EXPORT_COMPLETE: New training data exported
        - CONSOLIDATION_COMPLETE: Games consolidated into canonical DB

        Thread-safe: Can be called multiple times without re-subscribing.
        """
        if self._event_wired:
            return

        try:
            from app.coordination.event_router import get_event_router
            from app.coordination.data_events import DataEventType

            router = get_event_router()

            # Subscribe to events that affect data availability
            events_to_watch = [
                DataEventType.DATA_SYNC_COMPLETED,
                DataEventType.SELFPLAY_COMPLETE,
                DataEventType.NPZ_EXPORT_COMPLETE,
            ]

            # Try to add CONSOLIDATION_COMPLETE if it exists
            if hasattr(DataEventType, "CONSOLIDATION_COMPLETE"):
                events_to_watch.append(DataEventType.CONSOLIDATION_COMPLETE)

            for event_type in events_to_watch:
                router.subscribe(event_type, self._on_data_changed)

            self._event_wired = True
            logger.info("[GameDiscovery] Wired to data change events for cache invalidation")

        except ImportError:
            logger.debug("[GameDiscovery] Event router not available, skipping event wiring")
        except Exception as e:
            logger.warning(f"[GameDiscovery] Failed to wire events: {e}")

    def _on_data_changed(self, event: object) -> None:
        """Handle data change events by clearing cache.

        Args:
            event: Event object (DataEvent or similar)
        """
        try:
            # Extract event type for logging
            event_type = getattr(event, "event_type", type(event).__name__)
            logger.debug(f"[GameDiscovery] Cache invalidated by {event_type}")
            self.clear_cache()
        except Exception as e:
            logger.warning(f"[GameDiscovery] Error handling data change event: {e}")

    def _find_by_pattern(
        self, pattern: str, is_central: bool
    ) -> Iterator[DatabaseInfo]:
        """Find databases matching a pattern."""
        # Handle patterns with placeholders
        if "{board_type}" in pattern or "{num_players}" in pattern:
            for board_type in ALL_BOARD_TYPES:
                for num_players in ALL_PLAYER_COUNTS:
                    expanded = pattern.format(
                        board_type=board_type, num_players=num_players
                    )
                    yield from self._glob_pattern(
                        expanded, is_central, board_type, num_players, pattern
                    )
        else:
            yield from self._glob_pattern(pattern, is_central, None, None, pattern)

    def _glob_pattern(
        self,
        pattern: str,
        is_central: bool,
        board_type: str | None,
        num_players: int | None,
        source_pattern: str,
    ) -> Iterator[DatabaseInfo]:
        """Glob for databases matching a pattern."""
        full_pattern = self.root_path / pattern

        # Handle glob patterns
        if "*" in pattern:
            parent = full_pattern.parent
            while "*" in str(parent):
                parent = parent.parent
            if parent.exists():
                for match in self.root_path.glob(pattern):
                    if match.is_file() and match.stat().st_size > 0:
                        yield self._create_db_info(
                            match, is_central, board_type, num_players, source_pattern
                        )
        else:
            if full_pattern.exists() and full_pattern.stat().st_size > 0:
                yield self._create_db_info(
                    full_pattern, is_central, board_type, num_players, source_pattern
                )

    def _create_db_info(
        self,
        path: Path,
        is_central: bool,
        board_type: str | None,
        num_players: int | None,
        source_pattern: str,
    ) -> DatabaseInfo:
        """Create a DatabaseInfo object for a database."""
        # Try to infer board_type and num_players from path if not provided
        if board_type is None or num_players is None:
            inferred = self._infer_config_from_path(path)
            board_type = board_type or inferred[0]
            num_players = num_players or inferred[1]

        # Get game count and config breakdown
        game_count = 0
        config_counts: dict[tuple[str, int], int] = {}

        if is_central:
            # For central DBs, get breakdown by config
            config_counts = self._get_config_counts_dict(path)
            game_count = sum(config_counts.values())
        elif board_type and num_players:
            game_count = self._count_games_in_db(path, board_type, num_players)
            if game_count > 0:
                config_counts[(board_type, num_players)] = game_count

        return DatabaseInfo(
            path=path,
            board_type=board_type,
            num_players=num_players,
            game_count=game_count,
            is_central=is_central,
            source_pattern=source_pattern,
            config_counts=config_counts,
        )

    def _infer_config_from_path(self, path: Path) -> tuple[str | None, int | None]:
        """Try to infer board_type and num_players from the database path."""
        path_str = str(path).lower()

        # Check for board type
        board_type = None
        for bt in ALL_BOARD_TYPES:
            if bt in path_str:
                board_type = bt
                break

        # Check for player count
        num_players = None
        for np in ALL_PLAYER_COUNTS:
            if f"_{np}p" in path_str or f"_{np}/" in path_str or f"_{np}_" in path_str:
                num_players = np
                break

        return board_type, num_players

    def _validate_schema(self, conn: sqlite3.Connection) -> bool:
        """Validate database has required schema for game queries.

        Phase 5.3 Dec 29, 2025: Added to prevent silent 0-game returns
        when databases have incompatible schema.

        Dec 30, 2025: Added game_moves table validation to prevent export
        failures when databases have different move storage schemas.

        Args:
            conn: Open SQLite connection

        Returns:
            True if schema is valid, False otherwise
        """
        try:
            # Check if games table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
            )
            if not cursor.fetchone():
                return False

            # Check required columns exist in games table
            cursor = conn.execute("PRAGMA table_info(games)")
            columns = {row[1] for row in cursor.fetchall()}
            required = {"winner", "board_type", "num_players"}
            if not required.issubset(columns):
                return False

            # Check if game_moves table exists with required columns
            # Accept two schema formats:
            # 1. New format: move_json column (JSON-serialized move data)
            # 2. Legacy format: move_type + position_q + position_r columns
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
            )
            if cursor.fetchone():
                cursor = conn.execute("PRAGMA table_info(game_moves)")
                move_columns = {row[1] for row in cursor.fetchall()}
                # Accept either new (move_json) or legacy (move_type + positions) schema
                has_new_schema = "move_json" in move_columns
                has_legacy_schema = {"move_type", "position_q", "position_r"}.issubset(
                    move_columns
                )
                if not (has_new_schema or has_legacy_schema):
                    logger.debug(
                        f"game_moves table lacks required columns: "
                        f"need move_json or (move_type, position_q, position_r), "
                        f"found: {move_columns}"
                    )
                    return False

            return True
        except sqlite3.Error:
            return False

    def _count_games_in_db(
        self, db_path: Path, board_type: str, num_players: int
    ) -> int:
        """Count games for a specific config in a database."""
        if not db_path.exists():
            return 0

        # Get all aliases for this board type
        aliases = BOARD_TYPE_ALIASES.get(board_type, [board_type])
        placeholders = ",".join("?" * len(aliases))

        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
                # Phase 5.3: Validate schema before querying
                if not self._validate_schema(conn):
                    logger.debug(f"Schema validation failed for {db_path}")
                    return 0

                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM games WHERE winner IS NOT NULL "
                    f"AND board_type IN ({placeholders}) AND num_players = ?",
                    (*aliases, num_players),
                )
                return cursor.fetchone()[0]
        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Error counting games in {db_path}: {e}")
            return 0

    def _count_all_games(self, db_path: Path) -> int:
        """Count all completed games in a database."""
        if not db_path.exists():
            return 0

        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
                # Phase 5.3: Validate schema before querying
                if not self._validate_schema(conn):
                    logger.debug(f"Schema validation failed for {db_path}")
                    return 0

                cursor = conn.execute(
                    "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
                )
                return cursor.fetchone()[0]
        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Error counting games in {db_path}: {e}")
            return 0

    def _get_config_counts(self, db_path: Path) -> dict[str, int]:
        """Get game counts broken down by board_type and num_players (string keys)."""
        counts_dict = self._get_config_counts_dict(db_path)
        return {f"{bt}_{np}p": count for (bt, np), count in counts_dict.items()}

    def _get_config_counts_dict(self, db_path: Path) -> dict[tuple[str, int], int]:
        """Get game counts broken down by board_type and num_players (tuple keys)."""
        if not db_path.exists():
            return {}

        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
                # Phase 5.3: Validate schema before querying
                if not self._validate_schema(conn):
                    logger.debug(f"Schema validation failed for {db_path}")
                    return {}

                cursor = conn.execute(
                    "SELECT board_type, num_players, COUNT(*) "
                    "FROM games WHERE winner IS NOT NULL "
                    "GROUP BY board_type, num_players"
                )
                results = {}
                for row in cursor.fetchall():
                    board_type, num_players, count = row
                    if board_type and num_players:
                        results[(board_type, num_players)] = count
                return results
        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Error getting config counts from {db_path}: {e}")
            return {}

    # =========================================================================
    # JSONL Discovery Methods (December 2025)
    # =========================================================================

    def find_all_jsonl_files(self, use_cache: bool = True) -> list[JsonlFileInfo]:
        """Find all JSONL game files using known patterns.

        JSONL files contain game records from GPU selfplay that can be
        converted to NPZ format via scripts/jsonl_to_npz.py.

        Returns:
            List of JsonlFileInfo objects for all found JSONL files.
        """
        cache_key = "jsonl_all"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        jsonl_files = []
        seen_paths: set[Path] = set()

        for pattern, is_central in self.JSONL_PATTERNS:
            for jsonl_info in self._find_jsonl_by_pattern(pattern, is_central):
                if jsonl_info.path not in seen_paths:
                    seen_paths.add(jsonl_info.path)
                    jsonl_files.append(jsonl_info)

        self._cache[cache_key] = jsonl_files
        return jsonl_files

    def find_jsonl_for_config(
        self,
        board_type: str,
        num_players: int,
        use_cache: bool = True,
    ) -> list[JsonlFileInfo]:
        """Find JSONL files for a specific configuration.

        Args:
            board_type: Board type (square8, square19, hexagonal, hex8)
            num_players: Number of players (2, 3, 4)
            use_cache: Use cached results if available

        Returns:
            List of JsonlFileInfo objects matching the configuration.
        """
        cache_key = f"jsonl_{board_type}_{num_players}p"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        all_jsonl = self.find_all_jsonl_files(use_cache=use_cache)
        aliases = BOARD_TYPE_ALIASES.get(board_type, [board_type])

        matching = [
            j for j in all_jsonl
            if j.board_type in aliases and j.num_players == num_players
        ]

        self._cache[cache_key] = matching
        return matching

    def count_jsonl_games_by_config(self) -> dict[str, int]:
        """Count total games in JSONL files by configuration.

        Returns:
            Dict mapping config_key (e.g., "hex8_2p") to game count.
        """
        counts: dict[str, int] = {}
        for jsonl_info in self.find_all_jsonl_files():
            if jsonl_info.board_type and jsonl_info.num_players:
                config_key = f"{jsonl_info.board_type}_{jsonl_info.num_players}p"
                counts[config_key] = counts.get(config_key, 0) + jsonl_info.game_count
        return counts

    def _find_jsonl_by_pattern(
        self, pattern: str, is_central: bool
    ) -> Iterator[JsonlFileInfo]:
        """Find JSONL files matching a pattern.

        Args:
            pattern: Glob pattern with {board_type} and {num_players} placeholders
            is_central: Whether pattern matches multiple configs (not used for JSONL)

        Yields:
            JsonlFileInfo for each matching file
        """
        # Generate all board/player combinations
        for board_type in ALL_BOARD_TYPES:
            for num_players in ALL_PLAYER_COUNTS:
                concrete_pattern = pattern.replace(
                    "{board_type}", board_type
                ).replace(
                    "{num_players}", str(num_players)
                )

                for match in self.root_path.glob(concrete_pattern):
                    if match.is_file() and match.stat().st_size > 0:
                        yield JsonlFileInfo(
                            path=match,
                            board_type=board_type,
                            num_players=num_players,
                            source_pattern=pattern,
                        )

        # Also try pattern as-is (for catch-all patterns without placeholders)
        if "{board_type}" not in pattern and "{num_players}" not in pattern:
            for match in self.root_path.glob(pattern):
                if match.is_file() and match.stat().st_size > 0:
                    # Infer config from filename
                    board_type, num_players = self._infer_config_from_path(match)
                    yield JsonlFileInfo(
                        path=match,
                        board_type=board_type,
                        num_players=num_players,
                        source_pattern=pattern,
                    )


# Convenience functions for quick access
def find_all_game_databases(root_path: Path | str | None = None) -> list[DatabaseInfo]:
    """Find all game databases in the ai-service directory."""
    return GameDiscovery(root_path).find_all_databases()


def count_games_for_config(
    board_type: str, num_players: int, root_path: Path | str | None = None
) -> int:
    """Count total games for a specific board/player configuration."""
    return GameDiscovery(root_path).get_total_games(board_type, num_players)


def get_game_counts_summary(root_path: Path | str | None = None) -> dict[str, int]:
    """Get a summary of game counts by configuration."""
    return GameDiscovery(root_path).count_games_by_config().by_config


def get_game_counts_cluster_aware(root_path: Path | str | None = None) -> dict[str, int]:
    """Get game counts with cluster awareness.

    January 2026: Use this for health checks and status reporting on coordinator
    nodes that don't have local canonical databases.

    Tries UnifiedDataRegistry first (aggregates cluster + local + OWC + S3),
    falls back to local GameDiscovery if cluster data unavailable.

    Returns:
        Dict mapping config_key (e.g., 'hex8_2p') to total game count
    """
    # Try cluster-wide counts first
    try:
        from app.distributed.data_catalog import get_data_registry

        registry = get_data_registry()
        status = registry.get_cluster_status()
        if status and sum(v.get("total", 0) for v in status.values()) > 0:
            return {k: v.get("total", 0) for k, v in status.items()}
    except (ImportError, RuntimeError) as e:
        logger.debug(f"[get_game_counts_cluster_aware] Cluster registry unavailable: {e}")
    except Exception as e:
        logger.debug(f"[get_game_counts_cluster_aware] Error getting cluster counts: {e}")

    # Fall back to local discovery
    return get_game_counts_summary(root_path)


# JSONL convenience functions (December 2025)
def find_all_jsonl_files(root_path: Path | str | None = None) -> list[JsonlFileInfo]:
    """Find all JSONL game files in the ai-service directory."""
    return GameDiscovery(root_path).find_all_jsonl_files()


def find_jsonl_for_config(
    board_type: str, num_players: int, root_path: Path | str | None = None
) -> list[JsonlFileInfo]:
    """Find JSONL files for a specific board/player configuration."""
    return GameDiscovery(root_path).find_jsonl_for_config(board_type, num_players)


def get_jsonl_counts_summary(root_path: Path | str | None = None) -> dict[str, int]:
    """Get a summary of JSONL game counts by configuration."""
    return GameDiscovery(root_path).count_jsonl_games_by_config()


class RemoteGameDiscovery:
    """Discover games on remote hosts via SSH.

    Usage:
        remote = RemoteGameDiscovery()
        counts = remote.get_cluster_game_counts()
        print(counts)
        # {'gpu-node-1': {'hex8_2p': 35000, ...}, ...}
    """

    # Class-level cache for remote results
    _remote_cache: dict[str, tuple[dict[str, int], float]] = {}
    _cache_ttl: float = 300.0  # 5 minutes

    def __init__(self, hosts_config_path: Path | str | None = None, cache_ttl: float | None = None):
        """Initialize remote discovery.

        Args:
            hosts_config_path: Path to distributed_hosts.yaml
            cache_ttl: Cache TTL in seconds (default: 300, set to 0 to disable caching)
        """
        if cache_ttl is not None:
            self._cache_ttl = cache_ttl

        if hosts_config_path is None:
            # Auto-detect
            candidates = [
                Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml",
                Path.cwd() / "config" / "distributed_hosts.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    hosts_config_path = candidate
                    break

        self.hosts_config_path = Path(hosts_config_path) if hosts_config_path else None
        self._hosts: dict = {}
        self._load_hosts()

    def _load_hosts(self):
        """Load hosts from config file."""
        if not self.hosts_config_path or not self.hosts_config_path.exists():
            return

        try:
            import yaml
            with open(self.hosts_config_path) as f:
                config = yaml.safe_load(f) or {}
            self._hosts = config.get("hosts", {})
        except (FileNotFoundError, OSError, PermissionError, KeyError, AttributeError):
            pass
        except Exception as e:
            # Catch yaml.YAMLError and other YAML parsing errors
            if "yaml" in str(type(e).__module__).lower():
                pass
            else:
                raise

    def get_active_hosts(self) -> list[str]:
        """Get list of active host names."""
        return [
            name for name, info in self._hosts.items()
            if info.get("status") in ("ready", "active")
        ]

    def get_remote_game_counts(
        self,
        host_name: str,
        timeout: int = 30,
        use_cache: bool = True,
    ) -> dict[str, int]:
        """Get game counts from a remote host via SSH.

        Args:
            host_name: Name of the host in distributed_hosts.yaml
            timeout: SSH timeout in seconds
            use_cache: If True, use cached results if available and not expired

        Returns:
            Dict mapping config (e.g., 'hex8_2p') to game count
        """
        import subprocess
        import time

        # Check cache
        if use_cache and self._cache_ttl > 0:
            cached = self._remote_cache.get(host_name)
            if cached:
                cached_counts, cached_time = cached
                if time.time() - cached_time < self._cache_ttl:
                    logger.debug(f"Using cached results for {host_name}")
                    return cached_counts

        host_info = self._hosts.get(host_name, {})
        if not host_info:
            return {}

        ssh_host = host_info.get("tailscale_ip") or host_info.get("ssh_host")
        ssh_user = host_info.get("ssh_user", "ubuntu")
        ssh_key = host_info.get("ssh_key", "~/.ssh/id_cluster")
        ringrift_path = host_info.get("ringrift_path", "~/ringrift/ai-service")

        # Run game discovery on remote host
        cmd = [
            "ssh",
            "-i", os.path.expanduser(ssh_key),
            "-o", f"ConnectTimeout={timeout}",
            "-o", "StrictHostKeyChecking=no",
            f"{ssh_user}@{ssh_host}",
            f"cd {ringrift_path} && python3 -c \"from app.utils.game_discovery import get_game_counts_summary; import json; print(json.dumps(get_game_counts_summary()))\" 2>/dev/null",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
            if result.returncode == 0:
                import json
                # Parse JSON from output (handle welcome messages)
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("{"):
                        counts = json.loads(line)
                        # Cache the result
                        if self._cache_ttl > 0:
                            self._remote_cache[host_name] = (counts, time.time())
                        return counts
            return {}
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
            return {}
        except Exception as e:
            # Catch json.JSONDecodeError and other JSON parsing errors
            import json
            if isinstance(e, json.JSONDecodeError):
                return {}
            else:
                raise

    def clear_cache(self, host_name: str | None = None):
        """Clear cached remote results.

        Args:
            host_name: Specific host to clear, or None to clear all
        """
        if host_name:
            self._remote_cache.pop(host_name, None)
        else:
            self._remote_cache.clear()

    def get_cluster_game_counts(
        self,
        hosts: list[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, dict[str, int]]:
        """Get game counts from all cluster hosts.

        Args:
            hosts: List of host names to query (default: all active hosts)
            parallel: Query hosts in parallel

        Returns:
            Dict mapping host_name to {config: count} dict
        """
        if hosts is None:
            hosts = self.get_active_hosts()

        results = {}

        if parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.get_remote_game_counts, host): host
                    for host in hosts
                }
                for future in concurrent.futures.as_completed(futures):
                    host = futures[future]
                    try:
                        results[host] = future.result()
                    except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
                        results[host] = {}
                    except Exception as e:
                        # Catch json.JSONDecodeError from get_remote_game_counts
                        import json
                        if isinstance(e, json.JSONDecodeError):
                            results[host] = {}
                        else:
                            results[host] = {}
        else:
            for host in hosts:
                results[host] = self.get_remote_game_counts(host)

        return results

    def get_cluster_total_by_config(self) -> dict[str, int]:
        """Get total game counts across all hosts, grouped by config."""
        cluster_counts = self.get_cluster_game_counts()
        totals: dict[str, int] = {}

        for host_counts in cluster_counts.values():
            for config, count in host_counts.items():
                totals[config] = totals.get(config, 0) + count

        return totals


def run_diagnostics(discovery: GameDiscovery) -> None:
    """Run comprehensive diagnostics on game discovery."""
    print("\n" + "=" * 70)
    print("GAME DISCOVERY DIAGNOSTICS")
    print("=" * 70)

    print(f"\nRoot path: {discovery.root_path}")
    print(f"Root exists: {discovery.root_path.exists()}")

    # Check each pattern
    print("\n" + "-" * 70)
    print("PATTERN ANALYSIS")
    print("-" * 70)

    patterns_found = 0
    patterns_empty = 0
    all_dbs_by_pattern: dict[str, list[DatabaseInfo]] = {}

    for pattern, is_central in GameDiscovery.DB_PATTERNS:
        # Expand the pattern for each board type
        if "{board_type}" in pattern:
            for bt in ALL_BOARD_TYPES:
                for np in ALL_PLAYER_COUNTS:
                    expanded = pattern.format(board_type=bt, num_players=np)
                    matches = list(discovery._glob_pattern(expanded, is_central, bt, np, pattern))
                    if matches:
                        patterns_found += 1
                        key = f"{pattern} [{bt}_{np}p]"
                        all_dbs_by_pattern[key] = matches
                        print(f"  [OK] {expanded}")
                        for db in matches:
                            print(f"       -> {db.path.name}: {db.game_count:,} games")
        else:
            matches = list(discovery._glob_pattern(pattern, is_central, None, None, pattern))
            if matches:
                patterns_found += 1
                all_dbs_by_pattern[pattern] = matches
                print(f"  [OK] {pattern}")
                for db in matches:
                    print(f"       -> {db.path.name}: {db.game_count:,} games")
            else:
                patterns_empty += 1
                full_path = discovery.root_path / pattern.split("*")[0].rstrip("/")
                exists = full_path.exists() if "*" not in str(full_path) else "N/A"
                print(f"  [--] {pattern} (parent exists: {exists})")

    # Summary by config
    print("\n" + "-" * 70)
    print("COVERAGE BY CONFIGURATION")
    print("-" * 70)

    counts = discovery.count_games_by_config()
    for bt in ALL_BOARD_TYPES:
        for np in ALL_PLAYER_COUNTS:
            config = f"{bt}_{np}p"
            count = counts.by_config.get(config, 0)
            if count > 0:
                status = "OK"
            else:
                status = "MISSING"
            print(f"  [{status:7}] {config:20} {count:>10,} games")

    # Potential issues
    print("\n" + "-" * 70)
    print("POTENTIAL ISSUES")
    print("-" * 70)

    issues = []
    # Check for configs with no data
    for bt in ALL_BOARD_TYPES:
        for np in ALL_PLAYER_COUNTS:
            config = f"{bt}_{np}p"
            if counts.by_config.get(config, 0) == 0:
                issues.append(f"No games found for {config}")

    # Check for directories that might contain undiscovered databases
    potential_dirs = [
        "data/games",
        "data/selfplay",
        "data/training",
    ]
    for dir_path in potential_dirs:
        full_path = discovery.root_path / dir_path
        if full_path.exists():
            for db_file in full_path.rglob("*.db"):
                # Check if this DB was discovered
                found = any(
                    db.path == db_file
                    for dbs in all_dbs_by_pattern.values()
                    for db in dbs
                )
                if not found:
                    # Check if it has games
                    try:
                        with sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=2.0) as conn:
                            cursor = conn.execute("SELECT COUNT(*) FROM games")
                            game_count = cursor.fetchone()[0]
                            if game_count > 0:
                                issues.append(f"Undiscovered DB with {game_count:,} games: {db_file}")
                    except (sqlite3.Error, OSError, PermissionError):
                        pass

    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No issues found!")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Databases found: {counts.databases_found}")
    print(f"  Total games: {counts.total:,}")
    print(f"  Patterns with matches: {patterns_found}")
    print(f"  Patterns without matches: {patterns_empty}")
    print("=" * 70)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game Database Discovery")
    parser.add_argument("--root", type=str, help="Root path to ai-service")
    parser.add_argument(
        "--board-type", type=str, help="Filter by board type"
    )
    parser.add_argument(
        "--num-players", type=int, help="Filter by number of players"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--cluster", action="store_true", help="Query all cluster hosts")
    parser.add_argument("--diagnose", action="store_true", help="Run discovery diagnostics")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching for cluster queries")
    args = parser.parse_args()

    if args.diagnose:
        discovery = GameDiscovery(args.root)
        run_diagnostics(discovery)
    elif args.cluster:
        print("\nQuerying cluster hosts...")
        cache_ttl = 0 if args.no_cache else 300
        remote = RemoteGameDiscovery(cache_ttl=cache_ttl)
        cluster_counts = remote.get_cluster_game_counts()

        print("\nGame counts by host:")
        print("=" * 60)
        for host, counts in sorted(cluster_counts.items()):
            if counts:
                total = sum(counts.values())
                print(f"\n{host}: {total:,} total games")
                for config, count in sorted(counts.items()):
                    print(f"  {config}: {count:,}")
            else:
                print(f"\n{host}: (unreachable)")

        print("\n" + "=" * 60)
        totals = remote.get_cluster_total_by_config()
        print(f"\nCluster totals ({len([h for h in cluster_counts.values() if h])} hosts reachable):")
        for config, count in sorted(totals.items()):
            print(f"  {config}: {count:,}")
        print(f"\nGrand total: {sum(totals.values()):,} games")
    else:
        discovery = GameDiscovery(args.root)

        if args.board_type and args.num_players:
            dbs = discovery.find_databases_for_config(args.board_type, args.num_players)
            print(f"\nDatabases for {args.board_type} {args.num_players}p:")
            total = 0
            for db in dbs:
                print(f"  {db.path}: {db.game_count:,} games")
                total += db.game_count
            print(f"\nTotal: {total:,} games")
        else:
            counts = discovery.count_games_by_config()
            print(f"\nGame counts by configuration ({counts.databases_found} databases):")
            print("-" * 50)
            for config, count in sorted(counts.by_config.items()):
                print(f"  {config}: {count:,} games")
            print("-" * 50)
            print(f"Total: {counts.total:,} games")

            if args.verbose:
                print("\n\nAll databases found:")
                for db in discovery.find_all_databases():
                    print(f"  {db.path}")
                    print(f"    Pattern: {db.source_pattern}")
                    print(f"    Games: {db.game_count:,}")
                    print()
