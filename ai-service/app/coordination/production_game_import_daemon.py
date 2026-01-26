"""Production Game Import Daemon - Imports human vs AI games from ringrift.ai.

This daemon imports games played against AI on ringrift.ai (Sandbox and Lobby)
into the training pipeline. Human games are valuable training signal - especially
human wins which reveal AI weaknesses.

Workflow:
1. Periodically poll production HTTP endpoint for new human vs AI games
2. Parse JSONL stream and import games into staging database
3. Emit NEW_GAMES_AVAILABLE events to trigger DataConsolidationDaemon
4. Track last sync timestamp for incremental imports

This integrates with the existing daemon infrastructure:
- DataConsolidationDaemon: Handles merge into canonical databases
- DataPipelineOrchestrator: Handles NPZ export and training triggers
- Source weighting: Human games get 3x weight in training

Environment Variables:
    RINGRIFT_PRODUCTION_URL: Production server URL (default: https://ringrift.ai)
    RINGRIFT_PRODUCTION_SYNC_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_PRODUCTION_SYNC_INTERVAL: Check interval in seconds (default: 900)
    RINGRIFT_PRODUCTION_MIN_GAMES: Minimum games to trigger event (default: 10)

January 2026: Created as part of human game training infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.db.game_replay import GameReplayDB
from app.utils.canonical_naming import normalize_board_type as _canonical_normalize

logger = logging.getLogger(__name__)

__all__ = [
    "ProductionGameImportDaemon",
    "ProductionImportConfig",
    "get_production_game_import_daemon",
    "reset_production_game_import_daemon",
]


# ============================================================================
# Configuration
# ============================================================================

PRODUCTION_URL = os.getenv("RINGRIFT_PRODUCTION_URL", "https://ringrift.ai")

# Board type mapping (production uses enum values)
BOARD_TYPE_MAP = {
    "HEX8": "hex8",
    "HEXAGONAL": "hexagonal",
    "SQUARE8": "square8",
    "SQUARE19": "square19",
}


@dataclass
class ProductionImportConfig:
    """Configuration for Production Game Import daemon."""

    # Check interval (passed to HandlerBase as cycle_interval)
    check_interval_seconds: int = 900  # 15 minutes

    # Daemon control
    enabled: bool = True

    # Minimum games to trigger NEW_GAMES_AVAILABLE event
    min_games_for_event: int = 10

    # Maximum games to fetch per request
    fetch_limit: int = 500

    # Local staging directory for databases
    staging_dir: Path = field(default_factory=lambda: Path("data/games"))

    # Production connection
    production_url: str = PRODUCTION_URL

    # Request timeout
    request_timeout: int = 120  # 2 minutes for large responses

    @classmethod
    def from_env(cls) -> "ProductionImportConfig":
        """Load configuration from environment."""
        return cls(
            enabled=os.getenv("RINGRIFT_PRODUCTION_SYNC_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.getenv("RINGRIFT_PRODUCTION_SYNC_INTERVAL", "900")),
            min_games_for_event=int(os.getenv("RINGRIFT_PRODUCTION_MIN_GAMES", "10")),
            production_url=os.getenv("RINGRIFT_PRODUCTION_URL", PRODUCTION_URL),
        )


@dataclass
class ImportStats:
    """Statistics for an import cycle."""
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    games_fetched: int = 0
    games_imported: int = 0
    games_skipped: int = 0  # Duplicates
    configs_updated: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latest_timestamp: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.cycle_end - self.cycle_start


# ============================================================================
# Production Game Import Daemon
# ============================================================================


class ProductionGameImportDaemon(HandlerBase):
    """Daemon that imports human vs AI games from production.

    This daemon runs on the coordinator and periodically polls the production
    server for new games played against AI.

    January 2026: Created for human game training infrastructure.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking
    - State persisted in JSON file for incremental imports
    """

    def __init__(self, config: ProductionImportConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Configuration object, or None to use environment defaults.
        """
        self._config = config or ProductionImportConfig.from_env()

        super().__init__(
            name="production_game_import",
            cycle_interval=float(self._config.check_interval_seconds),
        )

        # State tracking
        self._last_sync_timestamp: str = ""
        self._state_file = Path("data/coordination/production_import_state.json")
        self._staging_db_path = self._config.staging_dir / "human_vs_ai_staging.db"

        # Stats
        self._stats["cycles_completed"] = 0
        self._stats["total_games_imported"] = 0
        self._stats["total_games_skipped"] = 0
        self._stats["errors"] = 0

        # Load persisted state
        self._load_state()

        # Initialize staging database
        self._staging_db: GameReplayDB | None = None

    def _load_state(self) -> None:
        """Load persisted state from JSON file."""
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    state = json.load(f)
                    self._last_sync_timestamp = state.get("last_sync_timestamp", "")
                    logger.info(
                        f"[ProductionImport] Loaded state: last_sync={self._last_sync_timestamp}"
                    )
        except Exception as e:
            logger.warning(f"[ProductionImport] Failed to load state: {e}")
            self._last_sync_timestamp = ""

    def _save_state(self) -> None:
        """Save state to JSON file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(
                    {
                        "last_sync_timestamp": self._last_sync_timestamp,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"[ProductionImport] Failed to save state: {e}")

    def _get_staging_db(self) -> GameReplayDB:
        """Get or create the staging database."""
        if self._staging_db is None:
            self._config.staging_dir.mkdir(parents=True, exist_ok=True)
            self._staging_db = GameReplayDB(str(self._staging_db_path))
        return self._staging_db

    async def _run_cycle(self) -> None:
        """Run one import cycle."""
        if not self._config.enabled:
            logger.debug("[ProductionImport] Daemon disabled, skipping cycle")
            return

        stats = ImportStats(cycle_start=time.time())

        try:
            # Fetch and import games
            await self._fetch_and_import_games(stats)

            # Emit events if we got enough new games
            if stats.games_imported >= self._config.min_games_for_event:
                self._emit_events(stats)

            # Save state for incremental imports
            if stats.latest_timestamp:
                self._last_sync_timestamp = stats.latest_timestamp
                self._save_state()

            # Update daemon stats
            self._stats["cycles_completed"] += 1
            self._stats["total_games_imported"] += stats.games_imported
            self._stats["total_games_skipped"] += stats.games_skipped

            stats.cycle_end = time.time()
            logger.info(
                f"[ProductionImport] Cycle completed in {stats.duration_seconds:.1f}s: "
                f"imported={stats.games_imported}, skipped={stats.games_skipped}, "
                f"configs={stats.configs_updated}"
            )

        except Exception as e:
            stats.errors.append(str(e))
            self._stats["errors"] += 1
            logger.error(f"[ProductionImport] Cycle failed: {e}")

    async def _fetch_and_import_games(self, stats: ImportStats) -> None:
        """Fetch games from production and import them."""
        try:
            import aiohttp
        except ImportError:
            logger.error("[ProductionImport] aiohttp not available")
            stats.errors.append("aiohttp not available")
            return

        url = f"{self._config.production_url}/api/training-export/human-games"
        params = {
            "limit": str(self._config.fetch_limit),
        }
        if self._last_sync_timestamp:
            params["since"] = self._last_sync_timestamp

        logger.info(
            f"[ProductionImport] Fetching games from {url} since={self._last_sync_timestamp}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self._config.request_timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        stats.errors.append(f"HTTP {resp.status}: {error_text[:200]}")
                        logger.error(
                            f"[ProductionImport] Failed to fetch: HTTP {resp.status}"
                        )
                        return

                    # Get total count from header
                    total_count = resp.headers.get("X-Total-Count", "?")
                    logger.info(f"[ProductionImport] Server reports {total_count} total games")

                    # Process JSONL stream
                    db = self._get_staging_db()
                    configs_seen: set[str] = set()

                    # Read entire response and split into lines
                    content = await resp.text()
                    lines = content.strip().split("\n") if content.strip() else []

                    for line in lines:
                        if not line.strip():
                            continue

                        stats.games_fetched += 1
                        try:
                            record = json.loads(line)
                            imported = self._import_game(db, record, stats)
                            if imported:
                                stats.games_imported += 1
                                # Track config
                                board_type = self._normalize_board_type(record.get("boardType", ""))
                                num_players = record.get("numPlayers", 2)
                                config_key = f"{board_type}_{num_players}p"
                                configs_seen.add(config_key)
                            else:
                                stats.games_skipped += 1

                            # Track latest timestamp
                            ended_at = record.get("endedAt", "")
                            if ended_at and ended_at > stats.latest_timestamp:
                                stats.latest_timestamp = ended_at

                        except json.JSONDecodeError as e:
                            logger.warning(f"[ProductionImport] Invalid JSON line: {e}")
                            stats.errors.append(f"JSON decode error: {e}")

                    stats.configs_updated = sorted(configs_seen)

        except aiohttp.ClientError as e:
            stats.errors.append(f"HTTP client error: {e}")
            logger.error(f"[ProductionImport] HTTP error: {e}")
        except asyncio.TimeoutError:
            stats.errors.append("Request timeout")
            logger.error("[ProductionImport] Request timed out")

    def _normalize_board_type(self, board_type: str) -> str:
        """Normalize board type from production format.

        Production uses uppercase enum names (HEX8, SQUARE8, etc.) which
        need to be converted to lowercase canonical values.

        January 2026: Delegates to canonical_naming for consistency.
        """
        # Try the direct mapping first (production uses uppercase)
        if board_type.upper() in BOARD_TYPE_MAP:
            return BOARD_TYPE_MAP[board_type.upper()]
        # Fall back to centralized normalization for other formats
        return _canonical_normalize(board_type)

    def _import_game(
        self, db: GameReplayDB, record: dict[str, Any], stats: ImportStats
    ) -> bool:
        """Import a single game record into the staging database.

        Args:
            db: The staging database.
            record: Game record from production.
            stats: Stats object for tracking.

        Returns:
            True if game was imported, False if skipped (duplicate).
        """
        game_id = record.get("id", "")
        if not game_id:
            logger.warning("[ProductionImport] Game record missing id")
            return False

        # Check for duplicate
        try:
            existing = db.get_game(game_id)
            if existing is not None:
                return False  # Already imported
        except Exception:
            pass  # Game doesn't exist, continue with import

        # Convert production format to our format
        board_type = self._normalize_board_type(record.get("boardType", ""))
        num_players = record.get("numPlayers", 2)

        # Extract metadata
        record_metadata = record.get("recordMetadata", {})
        if isinstance(record_metadata, str):
            try:
                record_metadata = json.loads(record_metadata)
            except json.JSONDecodeError:
                record_metadata = {}

        # Determine winner - map player position to 0-indexed
        winner = None
        if record.get("winnerId"):
            # Find which player position the winner was in
            for i in range(1, 5):
                player_key = f"player{i}Id"
                if record.get(player_key) == record.get("winnerId"):
                    winner = i - 1  # 0-indexed
                    break

        # Get initial state
        initial_state = record.get("initialState", {})
        if isinstance(initial_state, str):
            try:
                initial_state = json.loads(initial_state)
            except json.JSONDecodeError:
                initial_state = {}

        try:
            # Create game in staging database
            db.create_game(
                game_id=game_id,
                board_type=board_type,
                num_players=num_players,
                initial_state=initial_state,
                rng_seed=record.get("rngSeed"),
                source="human_vs_ai",
                metadata={
                    "humanPlayer": record_metadata.get("humanPlayer"),
                    "aiDifficulty": record_metadata.get("aiDifficulty"),
                    "aiType": record_metadata.get("aiType"),
                    "humanWon": record_metadata.get("humanWon"),
                    "importedFrom": "production",
                    "originalId": game_id,
                },
            )

            # Import moves
            moves = record.get("moves", [])
            for move_data in moves:
                self._import_move(db, game_id, move_data)

            # Mark game as complete if it has final state
            if record.get("finalState") and record.get("outcome"):
                termination = record.get("outcome", "completion")
                db.complete_game(
                    game_id=game_id,
                    winner=winner,
                    termination_reason=termination,
                )

            return True

        except Exception as e:
            logger.warning(f"[ProductionImport] Failed to import game {game_id}: {e}")
            stats.errors.append(f"Import error for {game_id}: {e}")
            return False

    def _import_move(
        self, db: GameReplayDB, game_id: str, move_data: dict[str, Any]
    ) -> None:
        """Import a single move into the database."""
        try:
            # Convert production move format to our format
            move_type = move_data.get("moveType", "").lower()
            player = move_data.get("playerIdx", 0)
            move_number = move_data.get("moveNumber", 0)

            # Build move dict in our format
            move = {
                "type": move_type,
                "player": player,
            }

            # Add position-specific fields based on move type
            if move_type in ("place_ring", "place_marker"):
                if move_data.get("toQ") is not None and move_data.get("toR") is not None:
                    move["to"] = {"q": move_data["toQ"], "r": move_data["toR"]}
            elif move_type == "move_stack":
                if move_data.get("fromQ") is not None and move_data.get("fromR") is not None:
                    move["from"] = {"q": move_data["fromQ"], "r": move_data["fromR"]}
                if move_data.get("toQ") is not None and move_data.get("toR") is not None:
                    move["to"] = {"q": move_data["toQ"], "r": move_data["toR"]}
            elif move_type == "choose_capture":
                if move_data.get("fromQ") is not None and move_data.get("fromR") is not None:
                    move["from"] = {"q": move_data["fromQ"], "r": move_data["fromR"]}
                if move_data.get("toQ") is not None and move_data.get("toR") is not None:
                    move["to"] = {"q": move_data["toQ"], "r": move_data["toR"]}

            # Add move to database
            db.add_move(
                game_id=game_id,
                move_number=move_number,
                player=player,
                move=move,
            )

        except Exception as e:
            logger.debug(f"[ProductionImport] Failed to import move: {e}")

    def _emit_events(self, stats: ImportStats) -> None:
        """Emit events after successful import."""
        # Emit per-config events
        for config_key in stats.configs_updated:
            safe_emit_event(
                "NEW_GAMES_AVAILABLE",
                {
                    "config_key": config_key,
                    "new_games": stats.games_imported,
                    "source": "human_vs_ai",
                    "origin": "production",
                },
            )

        # Emit summary event
        safe_emit_event(
            "DATA_SYNC_COMPLETED",
            {
                "sync_type": "production_human_games",
                "games_imported": stats.games_imported,
                "configs": stats.configs_updated,
                "duration_seconds": stats.duration_seconds,
            },
        )

        logger.info(
            f"[ProductionImport] Emitted events for {len(stats.configs_updated)} configs"
        )

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            HealthCheckResult for DaemonManager integration.
        """
        if not self._config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="Daemon disabled",
                details={"enabled": False},
            )

        # Check for recent errors
        error_count = self._stats.get("errors", 0)
        cycles = self._stats.get("cycles_completed", 0)

        # Determine health status
        if cycles == 0 and error_count > 0:
            status = CoordinatorStatus.ERROR
            message = "No successful cycles, errors occurred"
            healthy = False
        elif error_count > cycles * 0.5 and cycles > 0:  # >50% error rate
            status = CoordinatorStatus.DEGRADED
            message = f"High error rate: {error_count}/{cycles}"
            healthy = True
        else:
            status = CoordinatorStatus.RUNNING
            message = f"Healthy: {cycles} cycles, {self._stats.get('total_games_imported', 0)} games imported"
            healthy = True

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=message,
            details={
                "cycles_completed": cycles,
                "total_games_imported": self._stats.get("total_games_imported", 0),
                "total_games_skipped": self._stats.get("total_games_skipped", 0),
                "errors": error_count,
                "last_sync_timestamp": self._last_sync_timestamp,
                "production_url": self._config.production_url,
                "staging_db": str(self._staging_db_path),
            },
        )


# ============================================================================
# Singleton accessors
# ============================================================================

_instance: ProductionGameImportDaemon | None = None


def get_production_game_import_daemon() -> ProductionGameImportDaemon:
    """Get or create the singleton daemon instance."""
    global _instance
    if _instance is None:
        _instance = ProductionGameImportDaemon()
    return _instance


def reset_production_game_import_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    _instance = None
