"""DataPipelineManager: Data format conversion and consolidation.

January 2026: Extracted from p2p_orchestrator.py for better modularity.
Handles JSONL to DB/NPZ conversions and database consolidation.

This manager handles:
- Converting JSONL selfplay files to SQLite DB format
- Converting JSONL files directly to NPZ for training
- Consolidating siloed job databases into training DB
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from app.core.async_context import safe_create_task
from scripts.p2p.db_helpers import p2p_db_connection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: JSONL file opener (supports gzip)
# ============================================================================


def open_jsonl_file(path: Path):
    """Open a JSONL file, handling gzip compression."""
    import gzip

    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def safe_db_connection(db_path: Path, timeout: float = 5.0):
    """Context manager for safe SQLite database connection.

    Note: This is a thin wrapper around p2p_db_connection for backward compatibility.
    """
    return p2p_db_connection(db_path, timeout=timeout)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DataPipelineConfig:
    """Configuration for DataPipelineManager.

    Attributes:
        max_files_per_cycle: Maximum JSONL files to process per cycle
        chunk_size: Lines to read at a time from JSONL files
        large_backlog_threshold: Spawn external converter if more files
        jsonl_threshold_games: Only convert when > N games accumulated
        max_conversions_per_cycle: Limit NPZ conversions per cycle
        consolidation_cycle_interval: Run consolidation every N cycles
    """

    max_files_per_cycle: int = 50
    chunk_size: int = 500
    large_backlog_threshold: int = 200
    jsonl_threshold_games: int = 50
    max_conversions_per_cycle: int = 3
    consolidation_cycle_interval: int = 5


@dataclass
class DataPipelineStats:
    """Statistics for DataPipelineManager operations."""

    jsonl_to_db_conversions: int = 0
    jsonl_to_npz_conversions: int = 0
    games_converted_to_db: int = 0
    npz_files_created: int = 0
    db_consolidations: int = 0
    games_consolidated: int = 0
    last_conversion_time: float = 0.0
    last_consolidation_time: float = 0.0
    conversion_errors: int = 0
    consolidation_errors: int = 0


# ============================================================================
# Singleton management
# ============================================================================

_instance: DataPipelineManager | None = None


def get_data_pipeline_manager() -> DataPipelineManager | None:
    """Get the singleton DataPipelineManager instance."""
    return _instance


def set_data_pipeline_manager(manager: DataPipelineManager) -> None:
    """Set the singleton DataPipelineManager instance."""
    global _instance
    _instance = manager


def reset_data_pipeline_manager() -> None:
    """Reset the singleton DataPipelineManager instance (for testing)."""
    global _instance
    _instance = None


def create_data_pipeline_manager(
    config: DataPipelineConfig | None = None,
    orchestrator: Any | None = None,
) -> DataPipelineManager:
    """Create and register a DataPipelineManager instance.

    Args:
        config: Optional configuration
        orchestrator: P2P orchestrator reference (for callbacks)

    Returns:
        The created DataPipelineManager instance
    """
    manager = DataPipelineManager(config=config, orchestrator=orchestrator)
    set_data_pipeline_manager(manager)
    return manager


# ============================================================================
# DataPipelineManager
# ============================================================================


class DataPipelineManager:
    """Manager for data format conversion and consolidation.

    This class handles:
    - Converting JSONL selfplay files to SQLite DB format
    - Converting JSONL files directly to NPZ for training
    - Consolidating siloed job databases into training DB
    """

    def __init__(
        self,
        config: DataPipelineConfig | None = None,
        orchestrator: Any | None = None,
    ):
        """Initialize DataPipelineManager.

        Args:
            config: Configuration for the manager
            orchestrator: P2P orchestrator reference (for callbacks)
        """
        self.config = config or DataPipelineConfig()
        self._orchestrator = orchestrator
        self._stats = DataPipelineStats()
        self._consolidation_cycle = 0
        self._jsonl_aggregation_running = False
        self._npz_export_running = False

    @property
    def stats(self) -> DataPipelineStats:
        """Get current statistics."""
        return self._stats

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the P2P orchestrator reference.

        Called during orchestrator initialization.
        """
        self._orchestrator = orchestrator

    def _get_ai_service_path(self) -> str:
        """Get AI service path from orchestrator or environment."""
        if hasattr(self._orchestrator, "_get_ai_service_path"):
            return self._orchestrator._get_ai_service_path()
        return os.environ.get("AI_SERVICE_PATH", str(Path(__file__).parents[3]))

    def _get_node_selector(self) -> Any | None:
        """Get node selector from orchestrator."""
        return getattr(self._orchestrator, "node_selector", None)

    def _get_node_id(self) -> str:
        """Get this node's ID from orchestrator."""
        return getattr(self._orchestrator, "node_id", "unknown")

    # ========================================================================
    # Synchronous helper methods (run in thread pool)
    # ========================================================================

    def get_db_game_count_sync(self, db_path: Path) -> int:
        """Get game count from database synchronously.

        IMPORTANT: This is a blocking operation. Call via asyncio.to_thread() from async code.
        """
        try:
            with safe_db_connection(db_path, timeout=5) as conn:
                result = conn.execute("SELECT COUNT(*) FROM games").fetchone()
                return result[0] if result else 0
        except (sqlite3.Error, OSError):
            return 0

    def find_dbs_to_merge_sync(
        self, selfplay_dir: Path, main_db_path: Path
    ) -> list[tuple[Path, int]]:
        """Find databases that need merging synchronously.

        IMPORTANT: This is a blocking operation. Call via asyncio.to_thread() from async code.
        """
        dbs_to_merge = []
        for db_path in selfplay_dir.glob("**/games.db"):
            if ".tmp" in str(db_path) or db_path == main_db_path:
                continue
            try:
                with safe_db_connection(db_path, timeout=5) as conn:
                    count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                    if count > 0:
                        dbs_to_merge.append((db_path, count))
            except (KeyError, IndexError, AttributeError, sqlite3.Error):
                continue
        return dbs_to_merge

    # ========================================================================
    # JSONL to DB conversion
    # ========================================================================

    async def convert_jsonl_to_db(self, data_dir: Path, games_dir: Path) -> int:
        """Convert JSONL selfplay files to SQLite DB format for training.

        This enables the training pipeline to access games stored in JSONL format.
        Converted files are tracked to avoid re-processing.

        Features:
        - Chunked reading to handle large files without memory issues
        - Limited files per cycle to avoid blocking the event loop
        - Spawns external converter for large backlogs

        Returns:
            Number of games converted.
        """
        # Skip if disabled via environment variable
        if os.environ.get("RINGRIFT_SKIP_JSONL_CONVERSION", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.debug(
                "[DataPipelineManager] JSONL->DB conversion skipped via RINGRIFT_SKIP_JSONL_CONVERSION"
            )
            return 0

        config = self.config

        # Track converted files to avoid re-processing
        converted_marker_file = data_dir / ".jsonl_converted"
        converted_files: set = set()
        if converted_marker_file.exists():
            with contextlib.suppress(Exception):
                converted_files = set(
                    converted_marker_file.read_text().strip().split("\n")
                )

        total_converted = 0
        newly_converted = []
        selfplay_dir = data_dir / "selfplay"

        if not selfplay_dir.exists():
            return 0

        # Find all JSONL files in selfplay subdirectories
        jsonl_files = list(selfplay_dir.rglob("*.jsonl"))
        if not jsonl_files:
            return 0

        # Filter to unconverted files and sort by size (smallest first for quick wins)
        unconverted_files = []
        for jsonl_file in jsonl_files:
            try:
                file_size = jsonl_file.stat().st_size
                if file_size < 100:
                    continue
                file_key = str(jsonl_file.relative_to(data_dir))
                if file_key not in converted_files:
                    unconverted_files.append((jsonl_file, file_size, file_key))
            except (AttributeError, OSError):
                continue

        if not unconverted_files:
            return 0

        # Sort by size (smallest first) and limit per cycle
        unconverted_files.sort(key=lambda x: x[1])
        files_this_cycle = unconverted_files[: config.max_files_per_cycle]

        # If large backlog, spawn external converter in background
        if len(unconverted_files) > config.large_backlog_threshold:
            logger.info(
                f"Large JSONL backlog ({len(unconverted_files)} files), spawning background converter"
            )
            converter_script = (
                Path(self._get_ai_service_path())
                / "scripts"
                / "chunked_jsonl_converter.py"
            )
            if converter_script.exists():
                try:
                    subprocess.Popen(
                        [
                            "python3",
                            str(converter_script),
                            "--input-dir",
                            str(selfplay_dir),
                            "--output-dir",
                            str(games_dir),
                            "--workers",
                            "2",
                            "--chunk-size",
                            "500",
                        ],
                        stdout=open("/tmp/chunked_converter.log", "a"),
                        stderr=subprocess.STDOUT,
                        cwd=str(Path(self._get_ai_service_path())),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to spawn background converter: {e}")
                    self._stats.conversion_errors += 1

        # Group files by board type
        board_type_files: dict[str, list[tuple[Path, str]]] = {}
        for jsonl_file, _file_size, file_key in files_this_cycle:
            path_str = str(jsonl_file).lower()
            if "hex" in path_str:
                if "4p" in path_str:
                    board_key = "hexagonal_4p"
                elif "3p" in path_str:
                    board_key = "hexagonal_3p"
                else:
                    board_key = "hexagonal_2p"
            elif "square19" in path_str or "sq19" in path_str:
                if "4p" in path_str:
                    board_key = "square19_4p"
                elif "3p" in path_str:
                    board_key = "square19_3p"
                else:
                    board_key = "square19_2p"
            else:
                if "4p" in path_str:
                    board_key = "square8_4p"
                elif "3p" in path_str:
                    board_key = "square8_3p"
                else:
                    board_key = "square8_2p"

            if board_key not in board_type_files:
                board_type_files[board_key] = []
            board_type_files[board_key].append((jsonl_file, file_key))

        # Convert each board type to a consolidated DB (chunked reading)
        for board_key, files in board_type_files.items():
            if not files:
                continue

            games_added, converted = await asyncio.to_thread(
                self._convert_board_type_to_db,
                board_key,
                files,
                games_dir,
                config.chunk_size,
            )
            total_converted += games_added
            newly_converted.extend(converted)

        # Update converted files marker
        if newly_converted:
            try:
                all_converted = converted_files | set(newly_converted)
                converted_marker_file.write_text("\n".join(sorted(all_converted)))
            except (AttributeError, OSError):
                pass

        if total_converted > 0:
            self._stats.jsonl_to_db_conversions += 1
            self._stats.games_converted_to_db += total_converted
            self._stats.last_conversion_time = time.time()
            logger.info(
                f"JSONL conversion complete: {total_converted} total games converted"
            )

        return total_converted

    def _convert_board_type_to_db(
        self,
        board_key: str,
        files: list[tuple[Path, str]],
        games_dir: Path,
        chunk_size: int,
    ) -> tuple[int, list[str]]:
        """Convert one board type's JSONL files to SQLite DB.

        Runs in thread pool via asyncio.to_thread() to avoid blocking event loop.

        Returns:
            (games_added, newly_converted_file_keys)
        """
        db_path = games_dir / f"jsonl_converted_{board_key}.db"
        games_added = 0
        converted = []

        try:
          with p2p_db_connection(db_path, timeout=30.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT,
                    num_players INTEGER,
                    winner INTEGER,
                    move_count INTEGER,
                    game_status TEXT,
                    victory_type TEXT,
                    created_at TEXT,
                    source TEXT,
                    metadata_json TEXT
                )
            """
            )
            conn.commit()

            for jsonl_file, file_key in files:
                try:
                    # Read file in chunks to avoid memory issues
                    chunk_buffer = []
                    with open_jsonl_file(jsonl_file) as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                game_id = (
                                    f"{jsonl_file.stem}_{record.get('game_id', line_num)}"
                                )
                                chunk_buffer.append(
                                    (
                                        game_id,
                                        record.get(
                                            "board_type", board_key.split("_")[0]
                                        ),
                                        record.get(
                                            "num_players",
                                            int(board_key[-2])
                                            if board_key[-2].isdigit()
                                            else 2,
                                        ),
                                        record.get("winner", 0),
                                        record.get("move_count", 0),
                                        record.get("status", "completed"),
                                        record.get("victory_type", "unknown"),
                                        record.get("timestamp", ""),
                                        f"jsonl:{jsonl_file.name}",
                                        json.dumps(record),
                                    )
                                )

                                # Flush chunk when buffer is full
                                if len(chunk_buffer) >= chunk_size:
                                    conn.executemany(
                                        """
                                        INSERT OR IGNORE INTO games
                                        (game_id, board_type, num_players, winner, move_count,
                                         game_status, victory_type, created_at, source, metadata_json)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        chunk_buffer,
                                    )
                                    games_added += len(chunk_buffer)
                                    chunk_buffer = []

                            except (json.JSONDecodeError, Exception):
                                continue

                    # Flush remaining records
                    if chunk_buffer:
                        conn.executemany(
                            """
                            INSERT OR IGNORE INTO games
                            (game_id, board_type, num_players, winner, move_count,
                             game_status, victory_type, created_at, source, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            chunk_buffer,
                        )
                        games_added += len(chunk_buffer)

                    converted.append(file_key)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"converting {jsonl_file.name}: {e}")
                    continue

            conn.commit()

            if games_added > 0:
                logger.info(f"Converted {games_added} games to {db_path.name}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"creating DB for {board_key}: {e}")

        return games_added, converted

    # ========================================================================
    # JSONL to NPZ conversion
    # ========================================================================

    async def convert_jsonl_to_npz_for_training(
        self, data_dir: Path, training_dir: Path
    ) -> int:
        """Convert JSONL selfplay files directly to NPZ for training.

        This bypasses the DB intermediate step and creates training-ready NPZ files
        directly from JSONL. Called periodically when JSONL backlog exists.

        Returns:
            Number of NPZ files created.
        """
        # Skip if disabled via environment variable
        if os.environ.get("RINGRIFT_SKIP_JSONL_CONVERSION", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.debug(
                "[DataPipelineManager] JSONL->NPZ conversion skipped via RINGRIFT_SKIP_JSONL_CONVERSION"
            )
            return 0

        config = self.config
        selfplay_dir = data_dir / "selfplay"
        canonical_dir = selfplay_dir / "canonical"

        # Track converted files
        npz_marker_file = data_dir / ".jsonl_to_npz_converted"
        converted_files: set = set()
        if npz_marker_file.exists():
            with contextlib.suppress(Exception):
                converted_files = set(npz_marker_file.read_text().strip().split("\n"))

        conversions_done = 0
        newly_converted = []

        # Board configs to check
        board_configs = [
            ("square8", 2),
            ("square8", 3),
            ("square8", 4),
            ("square19", 2),
            ("square19", 3),
            ("square19", 4),
            ("hex8", 2),
            ("hex8", 3),
            ("hex8", 4),
            ("hexagonal", 2),
            ("hexagonal", 3),
            ("hexagonal", 4),
        ]

        for board_type, num_players in board_configs:
            if conversions_done >= config.max_conversions_per_cycle:
                break

            config_key = f"{board_type}_{num_players}p"

            # Skip if already converted recently
            if config_key in converted_files:
                continue

            # Find JSONL files for this config
            jsonl_files = []
            search_dirs = [canonical_dir, selfplay_dir, data_dir / "games"]

            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for jsonl_file in search_dir.rglob("*.jsonl"):
                    try:
                        if jsonl_file.stat().st_size < 100:
                            continue
                        jsonl_files.append(jsonl_file)
                    except (OSError, AttributeError):
                        continue

            if not jsonl_files:
                continue

            # Count games matching this config (quick check)
            game_count = 0
            valid_files = []
            for jsonl_file in jsonl_files:
                try:
                    with open_jsonl_file(jsonl_file) as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                game = json.loads(line)
                                game_board = game.get("board_type", "")
                                game_players = game.get("num_players", 0)
                                has_moves = "moves" in game and len(
                                    game.get("moves", [])
                                ) > 0

                                # Check if matches config
                                board_match = (
                                    board_type in game_board.lower()
                                    or game_board.lower() in board_type
                                )
                                if board_type == "hexagonal":
                                    board_match = "hex" in game_board.lower()

                                if (
                                    board_match
                                    and game_players == num_players
                                    and has_moves
                                ):
                                    game_count += 1
                                    if jsonl_file not in valid_files:
                                        valid_files.append(jsonl_file)
                            except json.JSONDecodeError:
                                continue
                except (AttributeError, OSError):
                    continue

            if game_count < config.jsonl_threshold_games:
                continue

            if not valid_files:
                continue

            # Convert to NPZ using jsonl_to_npz.py
            output_npz = training_dir / f"jsonl_auto_{config_key}_{int(time.time())}.npz"
            converter_script = (
                Path(self._get_ai_service_path()) / "scripts" / "jsonl_to_npz.py"
            )

            if not converter_script.exists():
                logger.info(f"JSONL->NPZ converter not found: {converter_script}")
                continue

            cmd = [
                sys.executable,
                str(converter_script),
                "--output",
                str(output_npz),
                "--board-type",
                board_type,
                "--num-players",
                str(num_players),
                "--sample-every",
                "5",
                "--max-games",
                "100",
            ]
            for vf in valid_files[:10]:  # Limit input files
                cmd.extend(["--input", str(vf)])

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(self._get_ai_service_path()))

            def _run_conversion():
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env,
                    cwd=str(Path(self._get_ai_service_path())),
                )

            try:
                logger.info(
                    f"Converting {game_count} {config_key} JSONL games to NPZ..."
                )
                result = await asyncio.to_thread(_run_conversion)
                if result.returncode == 0 and output_npz.exists():
                    logger.info(f"Created {output_npz.name} from JSONL")
                    conversions_done += 1
                    newly_converted.append(config_key)
                    self._stats.npz_files_created += 1
                else:
                    logger.info(
                        f"JSONL->NPZ conversion failed for {config_key}: "
                        f"{result.stderr[:200] if result.stderr else 'no error'}"
                    )
                    self._stats.conversion_errors += 1
            except subprocess.TimeoutExpired:
                logger.info(f"JSONL->NPZ conversion timeout for {config_key}")
                self._stats.conversion_errors += 1
            except Exception as e:  # noqa: BLE001
                logger.info(f"JSONL->NPZ conversion error for {config_key}: {e}")
                self._stats.conversion_errors += 1

        # Update marker file
        if newly_converted:
            try:
                all_converted = converted_files | set(newly_converted)
                npz_marker_file.write_text("\n".join(sorted(all_converted)))
            except (AttributeError, OSError):
                pass

        if conversions_done > 0:
            self._stats.jsonl_to_npz_conversions += conversions_done
            self._stats.last_conversion_time = time.time()

        return conversions_done

    # ========================================================================
    # Data consolidation
    # ========================================================================

    async def consolidate_selfplay_data(
        self,
        dispatch_export_job_callback: Callable | None = None,
    ) -> None:
        """Consolidate siloed job databases AND JSONL files into training DB.

        LEARNED LESSONS: Selfplay jobs write to job-specific databases for isolation.
        GPU selfplay jobs write JSONL files for efficiency.
        These need to be periodically merged into the training DB for:
        1. Training triggers to see accurate game counts
        2. Cross-node data sync to work correctly
        3. Training scripts to find all available data

        Runs every ~5 minutes (every 5th job check cycle) to avoid overhead.

        Args:
            dispatch_export_job_callback: Optional callback to dispatch export jobs
                to CPU nodes (called with node, input_path, output_path, etc.)
        """
        # Only run every N cycles (~5 minutes with JOB_CHECK_INTERVAL=60)
        self._consolidation_cycle += 1
        if self._consolidation_cycle % self.config.consolidation_cycle_interval != 0:
            return

        try:
            data_dir = Path(self._get_ai_service_path()) / "data"
            selfplay_dir = data_dir / "selfplay"
            games_dir = data_dir / "games"
            main_db_path = games_dir / "selfplay.db"
            jsonl_db_path = games_dir / "jsonl_aggregated.db"

            if not selfplay_dir.exists():
                return

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(self._get_ai_service_path()))

            # --- PART 1: Aggregate JSONL files (GPU selfplay output) ---
            jsonl_files = list(selfplay_dir.glob("**/games.jsonl"))
            # Filter to files > 1KB
            recent_jsonl = []
            for jf in jsonl_files:
                try:
                    if jf.stat().st_size > 1024:
                        recent_jsonl.append(jf)
                except (AttributeError, OSError):
                    pass

            if recent_jsonl:
                total_lines = 0
                for jf in recent_jsonl[:20]:  # Sample first 20
                    try:
                        with open(jf) as f:
                            total_lines += sum(1 for _ in f)
                    except (OSError,):
                        pass

                if total_lines > 100:  # Only run if there's meaningful data
                    aggregate_script = (
                        Path(self._get_ai_service_path())
                        / "scripts"
                        / "aggregate_jsonl_to_db.py"
                    )
                    if aggregate_script.exists() and not self._jsonl_aggregation_running:
                        self._jsonl_aggregation_running = True
                        logger.info(
                            f"JSONL aggregation: ~{total_lines * len(recent_jsonl) // 20} "
                            f"games in {len(recent_jsonl)} files"
                        )
                        cmd = [
                            sys.executable,
                            str(aggregate_script),
                            "--input-dir",
                            str(selfplay_dir),
                            "--output-db",
                            str(jsonl_db_path),
                        ]
                        proc = subprocess.Popen(
                            cmd,
                            env=env,
                            stdout=open("/tmp/jsonl_aggregate.log", "w"),
                            stderr=subprocess.STDOUT,
                            cwd=str(Path(self._get_ai_service_path())),
                        )
                        logger.info(f"Started JSONL aggregation (PID: {proc.pid})")
                        # Reset flag after ~10 minutes
                        asyncio.get_running_loop().call_later(
                            600, lambda: setattr(self, "_jsonl_aggregation_running", False)
                        )

            # --- PART 1b: Export NPZ from aggregated DB for training ---
            if jsonl_db_path.exists() and not self._npz_export_running:
                try:
                    game_count = await asyncio.to_thread(
                        self.get_db_game_count_sync, jsonl_db_path
                    )

                    # Only export if we have enough games and it's been a while
                    training_dir = data_dir / "training"
                    training_dir.mkdir(exist_ok=True)
                    npz_output = training_dir / "auto_training_sq8_2p.npz"

                    # Check if existing NPZ is stale (older than 1 hour) or small
                    should_export = False
                    if not npz_output.exists():
                        should_export = game_count > 500
                    else:
                        npz_age_hours = (
                            time.time() - npz_output.stat().st_mtime
                        ) / 3600
                        npz_size_mb = npz_output.stat().st_size / (1024 * 1024)
                        should_export = game_count > 1000 and (
                            npz_age_hours > 1 or npz_size_mb < 1
                        )

                    if should_export:
                        self._npz_export_running = True

                        # Find best CPU node for export (prefer vast nodes)
                        node_selector = self._get_node_selector()
                        export_node = None
                        node_id = self._get_node_id()

                        if node_selector:
                            cpu_nodes = node_selector.get_cpu_primary_nodes(count=3)
                            for node in cpu_nodes:
                                # Skip nodes that are already very loaded
                                if (
                                    node.get_load_score() < 80
                                    and node.cpu_percent < 90
                                ):
                                    export_node = node
                                    break

                        if (
                            export_node
                            and export_node.node_id != node_id
                            and dispatch_export_job_callback
                        ):
                            # Dispatch to high-CPU node
                            logger.info(
                                f"Dispatching NPZ export ({game_count} games) to "
                                f"{export_node.node_id} "
                                f"(cpu_power={export_node.cpu_power_score()}, "
                                f"cpus={export_node.cpu_count})"
                            )
                            safe_create_task(
                                dispatch_export_job_callback(
                                    node=export_node,
                                    input_path=str(jsonl_db_path),
                                    output_path=str(npz_output),
                                    board_type="square8",
                                    num_players=2,
                                    encoder_version="v3",
                                    max_games=5000,
                                    is_jsonl=False,
                                ),
                                name="pipeline-dispatch-export",
                            )
                        else:
                            # Fall back to local export if no suitable CPU node
                            export_script = (
                                Path(self._get_ai_service_path())
                                / "scripts"
                                / "export_replay_dataset.py"
                            )
                            if export_script.exists():
                                logger.info(
                                    f"Starting local NPZ export ({game_count} games) -> "
                                    f"{npz_output}"
                                )
                                cmd = [
                                    sys.executable,
                                    str(export_script),
                                    "--db",
                                    str(jsonl_db_path),
                                    "--board-type",
                                    "square8",
                                    "--num-players",
                                    "2",
                                    "--output",
                                    str(npz_output),
                                    "--encoder-version",
                                    "v3",
                                    "--max-games",
                                    "5000",
                                ]
                                subprocess.Popen(
                                    cmd,
                                    env=env,
                                    stdout=open("/tmp/npz_export.log", "w"),
                                    stderr=subprocess.STDOUT,
                                    cwd=str(Path(self._get_ai_service_path())),
                                )

                        # Reset flag after 30 minutes (export is slow)
                        asyncio.get_running_loop().call_later(
                            1800, lambda: setattr(self, "_npz_export_running", False)
                        )
                except Exception as e:  # noqa: BLE001
                    logger.info(f"NPZ export check error: {e}")
                    self._stats.consolidation_errors += 1

            # --- PART 2: Merge job DBs (CPU selfplay output) ---
            dbs_to_merge = await asyncio.to_thread(
                self.find_dbs_to_merge_sync, selfplay_dir, main_db_path
            )

            if dbs_to_merge:
                total_games = sum(c for _, c in dbs_to_merge)
                logger.info(
                    f"DB consolidation: {len(dbs_to_merge)} DBs with "
                    f"{total_games} games to merge"
                )

                # Use merge script if available
                merge_script = (
                    Path(self._get_ai_service_path()) / "scripts" / "merge_game_dbs.py"
                )
                if merge_script.exists():
                    # Build command with all DBs
                    cmd = [
                        sys.executable,
                        str(merge_script),
                        "--output",
                        str(main_db_path),
                        "--dedupe-by-game-id",
                    ]
                    for db_path, _ in dbs_to_merge[:50]:  # Limit to 50 DBs at a time
                        cmd.extend(["--db", str(db_path)])

                    # Run merge and wait for completion (capture output for debugging)
                    try:
                        result = await asyncio.to_thread(
                            subprocess.run,
                            cmd,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=600,  # 10 minute timeout
                            cwd=str(Path(self._get_ai_service_path())),
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"DB merge completed: {total_games} games from "
                                f"{len(dbs_to_merge)} DBs"
                            )
                            self._stats.db_consolidations += 1
                            self._stats.games_consolidated += total_games
                            self._stats.last_consolidation_time = time.time()
                        else:
                            logger.warning(
                                f"DB merge failed (rc={result.returncode}): "
                                f"{result.stderr[:500] if result.stderr else 'no stderr'}"
                            )
                            self._stats.consolidation_errors += 1
                    except subprocess.TimeoutExpired:
                        logger.warning("DB merge timed out after 600s")
                        self._stats.consolidation_errors += 1

        except Exception as e:  # noqa: BLE001
            logger.info(f"Data consolidation error: {e}")
            self._stats.consolidation_errors += 1

    # ========================================================================
    # Health check
    # ========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health check information for DaemonManager integration.

        Returns:
            Dict with health status and statistics
        """
        now = time.time()

        # Check conversion staleness
        time_since_last_conversion = (
            now - self._stats.last_conversion_time
            if self._stats.last_conversion_time > 0
            else float("inf")
        )
        time_since_last_consolidation = (
            now - self._stats.last_consolidation_time
            if self._stats.last_consolidation_time > 0
            else float("inf")
        )

        # Determine overall health
        if self._stats.conversion_errors > 10 or self._stats.consolidation_errors > 10:
            status = "DEGRADED"
        elif (
            time_since_last_conversion > 7200
            and time_since_last_consolidation > 7200
        ):
            status = "STALE"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "stats": {
                "jsonl_to_db_conversions": self._stats.jsonl_to_db_conversions,
                "jsonl_to_npz_conversions": self._stats.jsonl_to_npz_conversions,
                "games_converted_to_db": self._stats.games_converted_to_db,
                "npz_files_created": self._stats.npz_files_created,
                "db_consolidations": self._stats.db_consolidations,
                "games_consolidated": self._stats.games_consolidated,
                "conversion_errors": self._stats.conversion_errors,
                "consolidation_errors": self._stats.consolidation_errors,
            },
            "last_conversion_time": self._stats.last_conversion_time,
            "last_consolidation_time": self._stats.last_consolidation_time,
            "time_since_last_conversion": (
                time_since_last_conversion
                if self._stats.last_conversion_time > 0
                else None
            ),
            "time_since_last_consolidation": (
                time_since_last_consolidation
                if self._stats.last_consolidation_time > 0
                else None
            ),
            "jsonl_aggregation_running": self._jsonl_aggregation_running,
            "npz_export_running": self._npz_export_running,
        }
