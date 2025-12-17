#!/usr/bin/env python3
"""Comprehensive game statistics analysis tool for RingRift selfplay data.

This script analyzes selfplay game data across all board types and player counts,
producing detailed statistics on:
- Victory type distribution
- Win rates by player position
- Game length statistics
- Recovery action usage
- Board type comparisons
- Performance metrics
- AI type breakdown
- Data quality metrics

Usage:
    python scripts/analyze_game_statistics.py --data-dir data/selfplay
    python scripts/analyze_game_statistics.py --data-dir data/selfplay --output report.json
    python scripts/analyze_game_statistics.py --data-dir data/selfplay --format markdown

    # Recursively scan a directory tree for JSONL files:
    python scripts/analyze_game_statistics.py --jsonl-dir /path/to/synced/data --recursive

    # Quarantine bad/timeout data:
    python scripts/analyze_game_statistics.py --jsonl-dir data/games --quarantine-dir data/quarantine

    # Fix missing metadata in-place:
    python scripts/analyze_game_statistics.py --jsonl-dir data/games --fix-in-place

Output formats:
    - json: Machine-readable JSON report
    - markdown: Human-readable markdown report (default)
    - both: Both formats

Quarantine mode:
    When --quarantine-dir is specified, malformed records and timeout games are
    moved to separate files in the quarantine directory, organized by reason:
    - malformed/: Records that couldn't be parsed or normalized
    - timeout/: Games that ended due to timeout (config issues)
    - unknown_board/: Games where board type couldn't be determined

Fix-in-place mode:
    When --fix-in-place is specified, JSONL files are updated with normalized
    metadata (board_type, victory_type, num_players) where it was missing or
    in a non-standard format.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
AI_SERVICE_ROOT = Path(PROJECT_ROOT).resolve()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional canonical config imports (used only for drift detection). Keep this
# script usable in partial environments / older checkouts.
try:  # pragma: no cover
    from app.models.core import BoardType as _BoardType
    from app.rules.core import BOARD_CONFIGS as _CANON_BOARD_CONFIGS
    from app.rules.core import get_victory_threshold as _get_canon_victory_threshold

    _BOARD_TYPE_TO_ENUM = {
        "square8": _BoardType.SQUARE8,
        "square19": _BoardType.SQUARE19,
        "hexagonal": _BoardType.HEXAGONAL,
    }
    CANONICAL_RULES_AVAILABLE = True
except Exception:  # pragma: no cover
    _BOARD_TYPE_TO_ENUM = {}
    _CANON_BOARD_CONFIGS = {}
    _get_canon_victory_threshold = None  # type: ignore[assignment]
    CANONICAL_RULES_AVAILABLE = False


# =============================================================================
# Schema Normalization - handles old and new JSONL formats
# =============================================================================

# Mapping from board_size to board_type
BOARD_SIZE_TO_TYPE = {
    8: "square8",
    19: "square19",
    # Legacy hex embedding size used by early GPU/selfplay outputs.
    25: "hexagonal",
    11: "hexagonal",
    13: "hexagonal",
    15: "hexagonal",
}

# Mapping from old victory_type to normalized victory type
OLD_VICTORY_TYPE_MAP = {
    "ring_elimination": "elimination",
    "elimination": "elimination",
    "territory": "territory",
    "lps": "lps",
    "last_player_standing": "lps",
    "timeout": "timeout",
    "stalemate": "stalemate",
    "draw": "draw",
}

# Mapping from new termination_reason format to normalized victory type
TERMINATION_REASON_MAP = {
    "status:completed:elimination": "elimination",
    "status:completed:territory": "territory",
    "status:completed:lps": "lps",
    "status:completed:timeout": "timeout",
    "status:completed:stalemate": "stalemate",
    "status:completed:draw": "draw",
    "status:completed:unknown": "unknown",
}

# AI type inference from file paths
AI_PATH_PATTERNS = {
    "mcts_nn": "mcts+nn",
    "mcts-nn": "mcts+nn",
    "mcts_only": "mcts",
    "mcts-only": "mcts",
    "descent_nn": "descent+nn",
    "descent-nn": "descent+nn",
    "descent_only": "descent",
    "descent-only": "descent",
    "nn_only": "neural_net",
    "nn-only": "neural_net",
    "heuristic_only": "heuristic",
    "heuristic-only": "heuristic",
    "random_only": "random",
    "random-only": "random",
    "gpu_heuristic": "gpu_heuristic",
    "gpu_selfplay": "gpu_heuristic",
    "gpu_": "gpu_heuristic",
    "cpu_canonical": "cpu_heuristic",
    "fresh_cpu": "cpu_heuristic",
    "hybrid": "hybrid_gpu",
    # Host-based inference (from selfplay repository structure)
    "lambda-h100": "gpu_heuristic",
    "lambda-a10": "gpu_heuristic",
    "lambda-2xh100": "gpu_heuristic",
    "vast-5090": "gpu_heuristic",
    "vast-3090": "gpu_heuristic",
    "vast-3080": "gpu_heuristic",
    "vast-3070": "gpu_heuristic",
    "vast-3060": "gpu_heuristic",
    "mac-studio": "cpu_heuristic",
    "mbp-": "cpu_heuristic",
    "aws-": "cpu_heuristic",
    # Experiment types
    "canonical": "canonical_heuristic",
    "selfplay": "selfplay_heuristic",
    "soak": "soak_heuristic",
    "tournament": "tournament_mixed",
}

# Quarantine reasons
QUARANTINE_MALFORMED = "malformed"
QUARANTINE_TIMEOUT = "timeout"
QUARANTINE_UNKNOWN_BOARD = "unknown_board"
QUARANTINE_NO_WINNER = "no_winner"
QUARANTINE_ZERO_MOVES = "zero_moves"
QUARANTINE_ERROR_STATUS = "error_status"


def infer_ai_type(game: dict[str, Any], file_path: str = "") -> str:
    """Infer AI opponent type from game data or file path."""
    # Check explicit engine_mode first
    engine_mode = game.get("engine_mode")
    if engine_mode:
        mode_map = {
            "heuristic-only": "heuristic",
            "mcts-only": "mcts",
            "descent-only": "descent",
            "random-only": "random",
            "nn-only": "neural_net",
            "mixed": "mixed",
        }
        return mode_map.get(engine_mode, engine_mode)

    # Check ai_config or similar fields
    ai_config = game.get("ai_config") or game.get("ai_type") or game.get("opponent_type")
    if ai_config:
        return str(ai_config)

    # Infer from file path
    if file_path:
        path_lower = file_path.lower()
        for pattern, ai_type in AI_PATH_PATTERNS.items():
            if pattern in path_lower:
                return ai_type

    return "unknown"


def normalize_game(game: dict[str, Any], file_path: str = "") -> dict[str, Any]:
    """Normalize a game record to handle both old and new schema formats.

    Old schema fields:
      - board_size: int (8, 19, 25, etc.)
      - victory_type: "ring_elimination", "timeout", etc.
      - stalemate_tiebreaker: optional string

    New schema fields:
      - board_type: "square8", "square19", "hexagonal", etc.
      - termination_reason: "status:completed:elimination", etc.

    Returns a normalized game dict with consistent field names.
    Also tracks data quality via _inferred_* fields.
    """
    normalized = game.copy()
    # Track what was inferred vs explicit
    normalized["_inferred_board_type"] = False
    normalized["_inferred_num_players"] = False
    normalized["_inferred_victory_type"] = False
    normalized["_source_file"] = file_path

    # --- Normalize board_type ---
    if "board_type" not in normalized or normalized.get("board_type") == "unknown":
        normalized["_inferred_board_type"] = True
        if "config" in game and "board_type" in game["config"]:
            normalized["board_type"] = game["config"]["board_type"]
            normalized["_inferred_board_type"] = False  # Found in config
        elif "board_size" in game:
            board_size = game["board_size"]
            normalized["board_type"] = BOARD_SIZE_TO_TYPE.get(board_size, f"square{board_size}")
        elif "moves" in game and game["moves"]:
            max_coord = 0
            for move in game["moves"]:
                if isinstance(move, dict):
                    if "to" in move:
                        max_coord = max(max_coord, move["to"].get("x", 0), move["to"].get("y", 0))
            if max_coord <= 7:
                normalized["board_type"] = "square8"
            elif max_coord <= 18:
                normalized["board_type"] = "square19"
            else:
                normalized["board_type"] = f"square{max_coord + 1}"
        else:
            normalized["board_type"] = "unknown"
    elif normalized.get("board_type") == "square25":
        # Legacy alias used by some historical JSONLs / embeddings.
        normalized["board_type"] = "hexagonal"
        normalized["_inferred_board_type"] = True  # Normalized from legacy

    # --- Normalize num_players ---
    if "num_players" not in normalized:
        normalized["_inferred_num_players"] = True
        if "config" in game and "num_players" in game["config"]:
            normalized["num_players"] = game["config"]["num_players"]
            normalized["_inferred_num_players"] = False  # Found in config
        elif "moves" in game and game["moves"]:
            max_player = max(
                (m.get("player", 1) for m in game["moves"] if isinstance(m, dict) and "player" in m),
                default=2
            )
            normalized["num_players"] = max_player
        else:
            normalized["num_players"] = 2

    # --- Normalize victory_type ---
    # Prefer explicit victory_type when present; several generators historically
    # emitted termination_reason values that were too coarse (e.g. mapping all
    # non-threshold endings to "lps"). termination_reason remains a fallback for
    # older schemas that didn't emit victory_type.
    victory_type = None

    if "victory_type" in game and game["victory_type"] is not None:
        old_vtype = (
            game["victory_type"].lower()
            if isinstance(game["victory_type"], str)
            else str(game["victory_type"])
        )
        victory_type = OLD_VICTORY_TYPE_MAP.get(old_vtype, old_vtype)
    else:
        normalized["_inferred_victory_type"] = True

    if victory_type is None and "termination_reason" in game:
        tr = game["termination_reason"]
        if tr in TERMINATION_REASON_MAP:
            victory_type = TERMINATION_REASON_MAP[tr]
        elif isinstance(tr, str) and tr.startswith("status:completed:"):
            victory_type = tr.split(":")[-1]

    if victory_type == "stalemate" and game.get("stalemate_tiebreaker"):
        normalized["_stalemate_tiebreaker"] = game["stalemate_tiebreaker"].lower()

    if victory_type:
        normalized["victory_type"] = victory_type

    # --- Infer AI type ---
    normalized["_ai_type"] = infer_ai_type(game, file_path)

    # --- Determine quarantine reason (if any) ---
    quarantine_reason = None
    status = game.get("status") or game.get("game_status", "")
    move_count = game.get("move_count") or game.get("length") or len(game.get("moves", []))

    # Check for error statuses (e.g., error_reset from crashed selfplay)
    if status.startswith("error"):
        quarantine_reason = QUARANTINE_ERROR_STATUS
    # Check for zero-move games (except legitimate draws at initial state, which are impossible)
    elif move_count == 0:
        quarantine_reason = QUARANTINE_ZERO_MOVES
    elif normalized.get("board_type") == "unknown":
        quarantine_reason = QUARANTINE_UNKNOWN_BOARD
    elif victory_type == "timeout":
        quarantine_reason = QUARANTINE_TIMEOUT
    elif game.get("winner") is None and victory_type not in ("draw", "stalemate"):
        quarantine_reason = QUARANTINE_NO_WINNER
    normalized["_quarantine_reason"] = quarantine_reason

    return normalized


def is_completed_game(game: dict[str, Any], *, include_winner_only: bool) -> bool:
    """Check if a game record represents a completed game (not an eval pool position)."""
    if game.get("game_status") == "active":
        return False

    moves = game.get("moves", [])
    has_moves = isinstance(moves, list) and len(moves) > 0
    has_victory = "victory_type" in game or "termination_reason" in game
    has_winner = game.get("winner") is not None

    is_eval_pool = "move_history" in game and "moves" not in game
    if is_eval_pool:
        return False

    return has_moves or has_victory or (include_winner_only and has_winner)


@dataclass
class GameStats:
    """Statistics for a single game configuration (board + player count)."""

    board_type: str
    num_players: int
    total_games: int = 0
    total_moves: int = 0
    total_time_seconds: float = 0.0
    wins_by_player: dict[str, int] = field(default_factory=dict)
    victory_types: dict[str, int] = field(default_factory=dict)
    # Stalemate breakdown by tiebreaker type
    stalemate_by_tiebreaker: dict[str, int] = field(default_factory=dict)
    draws: int = 0
    games_with_recovery: int = 0
    total_recovery_opportunities: int = 0
    games_with_fe: int = 0
    game_lengths: list[int] = field(default_factory=list)  # Individual game lengths
    # Enhanced move analysis
    move_type_counts: dict[str, int] = field(default_factory=dict)
    # Recovery slide analysis
    games_with_recovery_slide: int = 0
    recovery_slides_by_player: dict[str, int] = field(default_factory=dict)  # player -> count
    wins_with_recovery_slide: int = 0  # Games where winner used recovery slide
    wins_without_recovery_slide: int = 0  # Games where winner did not use recovery slide
    # Capture chain analysis
    max_capture_chain_lengths: list[int] = field(default_factory=list)  # Max chain per game
    total_captures: int = 0
    total_chain_captures: int = 0
    # Forced elimination analysis
    fe_counts_per_game: list[int] = field(default_factory=list)
    fe_by_player: dict[str, int] = field(default_factory=dict)
    # Game phase analysis
    ring_placement_moves: int = 0  # How many place_ring moves
    territory_claims: int = 0  # process_territory_region moves
    line_formations: int = 0  # process_line moves
    # Tempo/momentum analysis
    consecutive_skips_by_player: dict[str, list[int]] = field(default_factory=dict)  # player -> list of skip streaks
    longest_skip_streak_per_game: list[int] = field(default_factory=list)
    # Comebacks: player in "losing" position who won
    games_with_late_fe_winner: int = 0  # Winner had forced elimination in game
    # Active game length (non-skip moves)
    active_moves_per_game: list[int] = field(default_factory=list)  # moves that aren't no_*_action
    # First blood analysis: who captures first
    first_capture_by_player: dict[str, int] = field(default_factory=dict)
    first_capturer_wins: int = 0
    first_capturer_loses: int = 0
    # Config/value sanity checks (helps detect rule drift in mixed datasets)
    starting_rings_per_player_counts: dict[str, int] = field(default_factory=dict)
    victory_threshold_counts: dict[str, int] = field(default_factory=dict)
    territory_victory_threshold_counts: dict[str, int] = field(default_factory=dict)
    lps_rounds_required_counts: dict[str, int] = field(default_factory=dict)
    # Joined breakdowns (rings/threshold -> outcomes), useful for mixed datasets.
    starting_rings_per_player_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    victory_threshold_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    games_with_starting_rings_mismatch: int = 0
    games_with_victory_threshold_mismatch: int = 0
    # Recovery mode breakdown (RR-CANON-R112)
    recovery_slides_by_mode: dict[str, int] = field(default_factory=dict)
    games_with_stack_strike: int = 0
    wins_with_stack_strike: int = 0
    # Timeout diagnostics
    timeout_move_count_hist: dict[str, int] = field(default_factory=dict)
    # AI type tracking
    games_by_ai_type: dict[str, int] = field(default_factory=dict)
    wins_by_ai_type: dict[str, dict[str, int]] = field(default_factory=dict)  # ai_type -> {player -> wins}
    victory_types_by_ai_type: dict[str, dict[str, int]] = field(default_factory=dict)  # ai_type -> {vtype -> count}
    # Data quality metrics
    games_with_inferred_board_type: int = 0
    games_with_inferred_num_players: int = 0
    games_with_inferred_victory_type: int = 0
    games_with_missing_winner: int = 0
    games_with_missing_moves: int = 0
    source_files: set[str] = field(default_factory=set)

    @property
    def moves_per_game(self) -> float:
        return self.total_moves / self.total_games if self.total_games > 0 else 0.0

    @property
    def games_per_second(self) -> float:
        return self.total_games / self.total_time_seconds if self.total_time_seconds > 0 else 0.0

    @property
    def min_moves(self) -> int:
        return min(self.game_lengths) if self.game_lengths else 0

    @property
    def max_moves(self) -> int:
        return max(self.game_lengths) if self.game_lengths else 0

    @property
    def median_moves(self) -> float:
        if not self.game_lengths:
            return 0.0
        sorted_lengths = sorted(self.game_lengths)
        n = len(sorted_lengths)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
        return float(sorted_lengths[mid])

    @property
    def std_moves(self) -> float:
        if len(self.game_lengths) < 2:
            return 0.0
        avg = self.moves_per_game
        variance = sum((x - avg) ** 2 for x in self.game_lengths) / len(self.game_lengths)
        return variance ** 0.5

    def win_rate(self, player: int) -> float:
        wins = self.wins_by_player.get(str(player), 0)
        return wins / self.total_games if self.total_games > 0 else 0.0

    def victory_type_rate(self, vtype: str) -> float:
        count = self.victory_types.get(vtype, 0)
        return count / self.total_games if self.total_games > 0 else 0.0

    def get_aggregated_victory_counts(self) -> dict[str, int]:
        """Get aggregated victory counts.

        Aggregation rules:
        - Territory: territory + stalemate (territory tiebreaker)
        - Elimination: elimination + ring_elimination + stalemate (ring_elimination tiebreaker)
        - LPS: lps (last player standing)
        """
        territory = self.victory_types.get("territory", 0)
        stalemate_territory = self.stalemate_by_tiebreaker.get("territory", 0)

        elimination = self.victory_types.get("elimination", 0)
        ring_elim = self.victory_types.get("ring_elimination", 0)
        stalemate_ring_elim = self.stalemate_by_tiebreaker.get("ring_elimination", 0)

        lps = self.victory_types.get("lps", 0)

        return {
            "territory": territory + stalemate_territory,
            "elimination": elimination + ring_elim + stalemate_ring_elim,
            "lps": lps,
        }


class QuarantineWriter:
    """Handles writing quarantined game records to organized output files."""

    def __init__(self, quarantine_dir: Path):
        self.quarantine_dir = quarantine_dir
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._file_handles: dict[str, Any] = {}
        self._counts: dict[str, int] = defaultdict(int)

    def write(self, game: dict[str, Any], reason: str, source_file: str) -> None:
        """Write a game record to the appropriate quarantine file."""
        reason_dir = self.quarantine_dir / reason
        reason_dir.mkdir(exist_ok=True)

        # Create a filename based on the source file
        source_name = Path(source_file).stem if source_file else "unknown"
        output_file = reason_dir / f"{source_name}.jsonl"

        # Remove internal tracking fields before writing
        clean_game = {k: v for k, v in game.items() if not k.startswith("_")}

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_game) + "\n")

        self._counts[reason] += 1

    def get_counts(self) -> dict[str, int]:
        """Return counts of quarantined records by reason."""
        return dict(self._counts)

    def close(self) -> None:
        """Close any open file handles."""
        pass  # Using context managers, so nothing to close


@dataclass
class DataQualityMetrics:
    """Aggregate data quality metrics across all games."""

    total_games_processed: int = 0
    games_with_explicit_board_type: int = 0
    games_with_inferred_board_type: int = 0
    games_with_explicit_num_players: int = 0
    games_with_inferred_num_players: int = 0
    games_with_explicit_victory_type: int = 0
    games_with_inferred_victory_type: int = 0
    games_with_winner: int = 0
    games_without_winner: int = 0
    games_with_moves: int = 0
    games_without_moves: int = 0
    quarantined_by_reason: dict[str, int] = field(default_factory=dict)
    malformed_records: int = 0
    games_by_source_host: dict[str, int] = field(default_factory=dict)


@dataclass
class CleanResult:
    """Result of cleaning a JSONL file."""

    total_records: int = 0
    kept_records: int = 0
    fixed_records: int = 0
    quarantined_by_reason: dict[str, int] = field(default_factory=dict)
    malformed_records: int = 0


@dataclass
class AnalysisReport:
    """Complete analysis report across all configurations."""

    stats_by_config: dict[tuple[str, int], GameStats] = field(default_factory=dict)
    recovery_analysis: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_sources: list[str] = field(default_factory=list)
    # AI type aggregation
    games_by_ai_type: dict[str, int] = field(default_factory=dict)
    # Data quality
    data_quality: DataQualityMetrics = field(default_factory=DataQualityMetrics)

    def total_games(self) -> int:
        return sum(s.total_games for s in self.stats_by_config.values())

    def total_moves(self) -> int:
        return sum(s.total_moves for s in self.stats_by_config.values())


def load_stats_json(path: Path) -> dict[str, Any] | None:
    """Load a stats.json file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return None


def load_recovery_analysis(path: Path) -> dict[str, Any] | None:
    """Load recovery analysis results if available."""
    candidates = [
        path / "recovery_analysis_results.json",
        path / "actual_recovery_opportunities.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _parse_game_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    # Support "Z" suffix.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _extract_game_timestamp_seconds(game: dict[str, Any]) -> float | None:
    # Prefer top-level timestamps when present (common in GPU selfplay JSONLs).
    for key in ("timestamp", "created_at", "generated_at", "start_time"):
        ts = _parse_game_timestamp(game.get(key))
        if ts is not None:
            return ts

    # Fallback to first move timestamp (common in soak JSONLs).
    moves = game.get("moves")
    if isinstance(moves, list) and moves:
        first = moves[0]
        if isinstance(first, dict):
            ts = _parse_game_timestamp(first.get("timestamp"))
            if ts is not None:
                return ts

    return None


def iter_jsonl_games(
    path: Path,
    *,
    include_winner_only: bool,
    game_cutoff_ts: float | None = None,
    include_unknown_game_timestamp: bool = False,
    quarantine_writer: QuarantineWriter | None = None,
    data_quality: DataQualityMetrics | None = None,
    exclude_quarantined: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield normalized completed games from a JSONL file.

    This is streaming by design to avoid loading large JSONL files into memory.

    Args:
        path: Path to the JSONL file
        include_winner_only: Include games with winner but no moves
        game_cutoff_ts: Only include games after this timestamp
        include_unknown_game_timestamp: Include games with unknown timestamps
        quarantine_writer: If provided, write quarantined records to this writer
        data_quality: If provided, update data quality metrics
        exclude_quarantined: If True, don't yield quarantined games
    """
    file_path_str = str(path)

    # Extract source host from path for data quality tracking
    source_host = "unknown"
    path_parts = str(path).lower().split("/")
    for part in path_parts:
        for pattern in ["lambda-", "vast-", "aws-", "mac-", "mbp-"]:
            if part.startswith(pattern):
                source_host = part
                break

    try:
        # Handle gzip-compressed files (by extension or magic bytes)
        is_gzip = str(path).endswith('.gz')
        if not is_gzip:
            # Check for gzip magic bytes (0x1f 0x8b)
            try:
                with open(path, 'rb') as check_f:
                    magic = check_f.read(2)
                    is_gzip = magic == b'\x1f\x8b'
            except Exception:
                pass

        if is_gzip:
            opener = gzip.open(path, "rt", encoding="utf-8")
        else:
            opener = open(path, "r", encoding="utf-8")

        with opener as f:
            line_num = 0
            while True:
                try:
                    line = f.readline()
                    if not line:
                        break
                    line_num += 1
                    line = line.strip()
                except (EOFError, gzip.BadGzipFile, OSError):
                    # Corrupted or truncated gzip file - stop reading
                    break
                if not line:
                    continue
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    if data_quality:
                        data_quality.malformed_records += 1
                    if quarantine_writer:
                        quarantine_writer.write(
                            {"_raw_line": line[:1000], "_line_num": line_num},
                            QUARANTINE_MALFORMED,
                            file_path_str,
                        )
                    continue

                if not is_completed_game(game, include_winner_only=include_winner_only):
                    continue

                if game_cutoff_ts is not None:
                    game_ts = _extract_game_timestamp_seconds(game)
                    if game_ts is None:
                        if not include_unknown_game_timestamp:
                            continue
                    elif game_ts < game_cutoff_ts:
                        continue

                normalized = normalize_game(game, file_path_str)

                # Update data quality metrics
                if data_quality:
                    data_quality.total_games_processed += 1
                    data_quality.games_by_source_host[source_host] = (
                        data_quality.games_by_source_host.get(source_host, 0) + 1
                    )
                    if normalized.get("_inferred_board_type"):
                        data_quality.games_with_inferred_board_type += 1
                    else:
                        data_quality.games_with_explicit_board_type += 1
                    if normalized.get("_inferred_num_players"):
                        data_quality.games_with_inferred_num_players += 1
                    else:
                        data_quality.games_with_explicit_num_players += 1
                    if normalized.get("_inferred_victory_type"):
                        data_quality.games_with_inferred_victory_type += 1
                    else:
                        data_quality.games_with_explicit_victory_type += 1
                    if game.get("winner") is not None:
                        data_quality.games_with_winner += 1
                    else:
                        data_quality.games_without_winner += 1
                    if game.get("moves"):
                        data_quality.games_with_moves += 1
                    else:
                        data_quality.games_without_moves += 1

                # Handle quarantine
                quarantine_reason = normalized.get("_quarantine_reason")
                if quarantine_reason:
                    if quarantine_writer:
                        quarantine_writer.write(normalized, quarantine_reason, file_path_str)
                    if data_quality:
                        data_quality.quarantined_by_reason[quarantine_reason] = (
                            data_quality.quarantined_by_reason.get(quarantine_reason, 0) + 1
                        )
                    if exclude_quarantined:
                        continue

                yield normalized
    except OSError:
        return


def fix_jsonl_in_place(
    path: Path,
    *,
    dry_run: bool = False,
    quiet: bool = False,
) -> tuple[int, int]:
    """Fix missing metadata in a JSONL file by normalizing all records.

    Args:
        path: Path to the JSONL file
        dry_run: If True, don't actually write changes
        quiet: Suppress progress messages

    Returns:
        Tuple of (total_records, modified_records)
    """
    file_path_str = str(path)
    records = []
    modified_count = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    records.append("")
                    continue
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    records.append(line)
                    continue

                normalized = normalize_game(game, file_path_str)

                # Check if any normalization was applied
                was_modified = False
                update_fields = ["board_type", "num_players", "victory_type"]
                for field in update_fields:
                    if field in normalized and normalized[field] != game.get(field):
                        was_modified = True
                        game[field] = normalized[field]

                if was_modified:
                    modified_count += 1

                records.append(json.dumps(game))

    except OSError as e:
        if not quiet:
            print(f"Error reading {path}: {e}", file=sys.stderr)
        return 0, 0

    if modified_count > 0 and not dry_run:
        try:
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(record + "\n")
            if not quiet:
                print(f"Fixed {modified_count} records in {path}", file=sys.stderr)
        except OSError as e:
            if not quiet:
                print(f"Error writing {path}: {e}", file=sys.stderr)

    return len([r for r in records if r]), modified_count


def clean_jsonl_file(
    path: Path,
    quarantine_dir: Path | None = None,
    *,
    dry_run: bool = False,
    quiet: bool = False,
) -> CleanResult:
    """Clean a JSONL file: fix metadata and quarantine bad records.

    This function:
    1. Normalizes metadata (board_type, num_players, victory_type)
    2. Removes quarantinable records (timeout, unknown_board, no_winner, malformed)
    3. Writes quarantined records to quarantine_dir (if provided)
    4. Rewrites the original file with only good, normalized records

    Args:
        path: Path to the JSONL file
        quarantine_dir: Directory to write quarantined records (optional)
        dry_run: If True, don't actually write changes
        quiet: Suppress progress messages

    Returns:
        CleanResult with counts of processed/kept/fixed/quarantined records
    """
    file_path_str = str(path)
    result = CleanResult()

    good_records: list[str] = []
    quarantined: dict[str, list[dict[str, Any]]] = {}

    try:
        # Use errors="replace" to handle files with encoding issues (corrupted bytes)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                result.total_records += 1

                # Try to parse
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    result.malformed_records += 1
                    quarantined.setdefault("malformed", []).append({"_raw_line": line})
                    result.quarantined_by_reason["malformed"] = result.quarantined_by_reason.get("malformed", 0) + 1
                    continue

                # Normalize the game
                normalized = normalize_game(game, file_path_str)

                # Check if normalization changed anything
                was_fixed = False
                for field in ["board_type", "num_players", "victory_type"]:
                    if field in normalized and normalized[field] != game.get(field):
                        was_fixed = True
                        game[field] = normalized[field]

                if was_fixed:
                    result.fixed_records += 1

                # Check quarantine conditions
                quarantine_reason: str | None = None

                # Check for error status (e.g., error_reset from crashed selfplay)
                status = game.get("status") or game.get("game_status", "")
                if isinstance(status, str) and status.startswith("error"):
                    quarantine_reason = "error_status"

                # Check for zero-move games
                move_count = game.get("move_count") or game.get("length") or 0
                moves = game.get("moves") or game.get("move_history") or []
                if quarantine_reason is None and move_count == 0 and len(moves) == 0:
                    quarantine_reason = "zero_moves"

                # Check for unknown board type
                board_type = game.get("board_type")
                if quarantine_reason is None and (board_type is None or board_type == "unknown"):
                    quarantine_reason = "unknown_board"

                # Check for timeout/no winner
                termination = game.get("termination_reason") or game.get("termination")
                if quarantine_reason is None:
                    if termination == "timeout":
                        quarantine_reason = "timeout"
                    elif termination in ("no_winner", "draw"):
                        quarantine_reason = "no_winner"

                # Check for missing winner (for completed games)
                if quarantine_reason is None:
                    winner = game.get("winner")
                    if winner is None and len(moves) > 0:
                        quarantine_reason = "no_winner"

                if quarantine_reason:
                    quarantined.setdefault(quarantine_reason, []).append(game)
                    result.quarantined_by_reason[quarantine_reason] = result.quarantined_by_reason.get(quarantine_reason, 0) + 1
                else:
                    good_records.append(json.dumps(game))
                    result.kept_records += 1

    except OSError as e:
        if not quiet:
            print(f"Error reading {path}: {e}", file=sys.stderr)
        return result

    # Write quarantined records
    if quarantine_dir and quarantined and not dry_run:
        for reason, records in quarantined.items():
            reason_dir = quarantine_dir / reason
            reason_dir.mkdir(parents=True, exist_ok=True)
            quarantine_file = reason_dir / path.name
            try:
                with open(quarantine_file, "a", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")
            except OSError as e:
                if not quiet:
                    print(f"Error writing quarantine file {quarantine_file}: {e}", file=sys.stderr)

    # Rewrite original file with only good records
    if not dry_run and result.total_records > 0:
        try:
            with open(path, "w", encoding="utf-8") as f:
                for record in good_records:
                    f.write(record + "\n")
            if not quiet and (result.fixed_records > 0 or sum(result.quarantined_by_reason.values()) > 0):
                q_total = sum(result.quarantined_by_reason.values())
                print(f"Cleaned {path.name}: {result.kept_records} kept, {result.fixed_records} fixed, {q_total} quarantined", file=sys.stderr)
        except OSError as e:
            if not quiet:
                print(f"Error writing {path}: {e}", file=sys.stderr)

    return result


def _get_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _extract_starting_rings_per_player(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        players = state.get("players")
        if isinstance(players, list) and players:
            rings: list[int] = []
            for p in players:
                if not isinstance(p, dict):
                    continue
                value = _get_first(p, ["rings_in_hand", "ringsInHand"])
                if isinstance(value, int):
                    rings.append(value)
            if rings:
                if len(set(rings)) == 1:
                    return rings[0]
                return min(rings)

        # Some traces store a matrix of rings-in-hand by player index.
        matrix = _get_first(state, ["rings_in_hand", "ringsInHand"])
        if isinstance(matrix, list) and matrix:
            ints = [v for v in matrix if isinstance(v, int)]
            if ints:
                if len(set(ints)) == 1:
                    return ints[0]
                return min(ints)

    config = game.get("config")
    if isinstance(config, dict):
        value = _get_first(config, ["rings_per_player", "ringsPerPlayer"])
        if isinstance(value, int):
            return value

    return None


def _extract_victory_threshold(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        value = _get_first(state, ["victory_threshold", "victoryThreshold"])
        if isinstance(value, int):
            return value
    return None


def _extract_territory_victory_threshold(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        value = _get_first(state, ["territory_victory_threshold", "territoryVictoryThreshold"])
        if isinstance(value, int):
            return value
    return None


def _extract_lps_rounds_required(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        rules_options = _get_first(state, ["rules_options", "rulesOptions"])
        if isinstance(rules_options, dict):
            value = _get_first(rules_options, ["lpsRoundsRequired", "lps_rounds_required"])
            if isinstance(value, int):
                return value
    return None


def collect_stats_from_jsonl(
    jsonl_files: list[Path],
    report: AnalysisReport,
    *,
    include_winner_only: bool,
    game_cutoff_ts: float | None = None,
    include_unknown_game_timestamp: bool = False,
    quarantine_writer: QuarantineWriter | None = None,
    data_quality: DataQualityMetrics | None = None,
    exclude_quarantined: bool = True,
    ai_types_filter: set[str] | None = None,
) -> None:
    """Collect statistics from JSONL files and add to report."""
    for jsonl_path in jsonl_files:
        any_games = False
        for game in iter_jsonl_games(
            jsonl_path,
            include_winner_only=include_winner_only,
            game_cutoff_ts=game_cutoff_ts,
            include_unknown_game_timestamp=include_unknown_game_timestamp,
            quarantine_writer=quarantine_writer,
            data_quality=data_quality,
            exclude_quarantined=exclude_quarantined,
        ):
            # Filter by AI type if specified
            if ai_types_filter:
                game_ai_type = game.get("_ai_type", "unknown")
                if game_ai_type not in ai_types_filter:
                    continue
            any_games = True
            board_type = game.get("board_type", "unknown")
            num_players = int(game.get("num_players", 2) or 2)
            key = (board_type, num_players)
            if key not in report.stats_by_config:
                report.stats_by_config[key] = GameStats(board_type=board_type, num_players=num_players)

            stats = report.stats_by_config[key]

            expected_rings: int | None = None
            expected_victory_threshold: int | None = None
            if CANONICAL_RULES_AVAILABLE and board_type in _BOARD_TYPE_TO_ENUM:
                bt = _BOARD_TYPE_TO_ENUM[board_type]
                cfg = _CANON_BOARD_CONFIGS.get(bt)
                if cfg is not None:
                    expected_rings = int(cfg.rings_per_player)
                if _get_canon_victory_threshold is not None:
                    try:
                        expected_victory_threshold = int(_get_canon_victory_threshold(bt, num_players))
                    except Exception:
                        expected_victory_threshold = None

            stats.total_games += 1
            move_count = game.get("move_count", len(game.get("moves", [])))
            stats.total_moves += move_count
            stats.game_lengths.append(move_count)
            stats.total_time_seconds += game.get("game_time_seconds", 0.0)

            # Initial-state sanity checks / drift detection.
            starting_rings = _extract_starting_rings_per_player(game)
            starting_rings_key: str | None = None
            if starting_rings is not None:
                starting_rings_key = str(starting_rings)
                stats.starting_rings_per_player_counts[starting_rings_key] = (
                    stats.starting_rings_per_player_counts.get(starting_rings_key, 0) + 1
                )
                if expected_rings is not None and starting_rings != expected_rings:
                    stats.games_with_starting_rings_mismatch += 1

            victory_threshold = _extract_victory_threshold(game)
            victory_threshold_key: str | None = None
            if victory_threshold is not None:
                victory_threshold_key = str(victory_threshold)
                stats.victory_threshold_counts[victory_threshold_key] = (
                    stats.victory_threshold_counts.get(victory_threshold_key, 0) + 1
                )
                if expected_victory_threshold is not None and victory_threshold != expected_victory_threshold:
                    stats.games_with_victory_threshold_mismatch += 1

            territory_threshold = _extract_territory_victory_threshold(game)
            if territory_threshold is not None:
                key_str = str(territory_threshold)
                stats.territory_victory_threshold_counts[key_str] = (
                    stats.territory_victory_threshold_counts.get(key_str, 0) + 1
                )

            lps_rounds_required = _extract_lps_rounds_required(game)
            if lps_rounds_required is not None:
                key_str = str(lps_rounds_required)
                stats.lps_rounds_required_counts[key_str] = (
                    stats.lps_rounds_required_counts.get(key_str, 0) + 1
                )

            # Winner
            winner = game.get("winner")
            if winner is not None:
                stats.wins_by_player[str(winner)] = stats.wins_by_player.get(str(winner), 0) + 1

            # Victory type
            victory_type = game.get("victory_type") or "unknown"
            stats.victory_types[victory_type] = stats.victory_types.get(victory_type, 0) + 1
            if victory_type == "timeout":
                move_key = str(int(move_count) if isinstance(move_count, int) else 0)
                stats.timeout_move_count_hist[move_key] = (
                    stats.timeout_move_count_hist.get(move_key, 0) + 1
                )

            # AI type tracking
            ai_type = game.get("_ai_type", "unknown")
            stats.games_by_ai_type[ai_type] = stats.games_by_ai_type.get(ai_type, 0) + 1
            if winner is not None:
                if ai_type not in stats.wins_by_ai_type:
                    stats.wins_by_ai_type[ai_type] = {}
                stats.wins_by_ai_type[ai_type][str(winner)] = (
                    stats.wins_by_ai_type[ai_type].get(str(winner), 0) + 1
                )
            if ai_type not in stats.victory_types_by_ai_type:
                stats.victory_types_by_ai_type[ai_type] = {}
            stats.victory_types_by_ai_type[ai_type][victory_type] = (
                stats.victory_types_by_ai_type[ai_type].get(victory_type, 0) + 1
            )
            # Report-level AI type aggregation
            report.games_by_ai_type[ai_type] = report.games_by_ai_type.get(ai_type, 0) + 1

            # Data quality tracking (per-config)
            if game.get("_inferred_board_type"):
                stats.games_with_inferred_board_type += 1
            if game.get("_inferred_num_players"):
                stats.games_with_inferred_num_players += 1
            if game.get("_inferred_victory_type"):
                stats.games_with_inferred_victory_type += 1
            if winner is None:
                stats.games_with_missing_winner += 1
            if not game.get("moves"):
                stats.games_with_missing_moves += 1
            source_file = game.get("_source_file", "")
            if source_file:
                stats.source_files.add(source_file)

            # Joined breakdowns (rings/threshold -> outcomes).
            if starting_rings_key is not None:
                entry = stats.starting_rings_per_player_breakdown.setdefault(
                    starting_rings_key,
                    {"games": 0, "total_moves": 0, "victory_types": {}},
                )
                entry["games"] = int(entry.get("games", 0) or 0) + 1
                entry["total_moves"] = int(entry.get("total_moves", 0) or 0) + int(move_count or 0)
                vt = entry.setdefault("victory_types", {})
                vt[victory_type] = int(vt.get(victory_type, 0) or 0) + 1

            if victory_threshold_key is not None:
                entry = stats.victory_threshold_breakdown.setdefault(
                    victory_threshold_key,
                    {"games": 0, "total_moves": 0, "victory_types": {}},
                )
                entry["games"] = int(entry.get("games", 0) or 0) + 1
                entry["total_moves"] = int(entry.get("total_moves", 0) or 0) + int(move_count or 0)
                vt = entry.setdefault("victory_types", {})
                vt[victory_type] = int(vt.get(victory_type, 0) or 0) + 1

            # Stalemate tiebreaker
            tiebreaker = game.get("stalemate_tiebreaker")
            if tiebreaker and victory_type == "stalemate":
                stats.stalemate_by_tiebreaker[tiebreaker] = (
                    stats.stalemate_by_tiebreaker.get(tiebreaker, 0) + 1
                )

            # Analyze moves in detail
            moves = game.get("moves", [])
            has_fe = False
            fe_count = 0
            has_recovery_slide = False
            has_stack_strike = False
            recovery_slides_by_player_in_game: dict[str, int] = {}
            stack_strikes_by_player_in_game: set[str] = set()
            fe_by_player_in_game: dict[str, int] = {}
            capture_chain_length = 0
            max_chain_in_game = 0
            in_chain = False
            first_capture_player = None
            active_move_count = 0
            # Track consecutive skips per player
            current_skip_streak: dict[str, int] = {}
            max_skip_streak = 0

            for m in moves:
                if not isinstance(m, dict):
                    continue
                move_type = m.get("type", "")
                player = m.get("player")
                player_str = str(player) if player is not None else None

                # Count move types
                stats.move_type_counts[move_type] = stats.move_type_counts.get(move_type, 0) + 1

                # Track phase-specific actions
                if move_type == "place_ring":
                    stats.ring_placement_moves += 1
                elif move_type in ("choose_territory_option", "process_territory_region"):
                    stats.territory_claims += 1
                elif move_type in ("choose_line_option", "choose_line_reward", "process_line"):
                    stats.line_formations += 1

                # Track active vs skip moves
                is_skip = move_type.startswith("no_") or move_type.startswith("skip_")
                if not is_skip:
                    active_move_count += 1
                    # Reset skip streak for this player
                    if player_str:
                        if player_str in current_skip_streak and current_skip_streak[player_str] > 0:
                            if player_str not in stats.consecutive_skips_by_player:
                                stats.consecutive_skips_by_player[player_str] = []
                            stats.consecutive_skips_by_player[player_str].append(current_skip_streak[player_str])
                        current_skip_streak[player_str] = 0
                elif player_str:
                    # Increment skip streak
                    current_skip_streak[player_str] = current_skip_streak.get(player_str, 0) + 1
                    if current_skip_streak[player_str] > max_skip_streak:
                        max_skip_streak = current_skip_streak[player_str]

                # Track forced elimination
                if move_type == "forced_elimination":
                    has_fe = True
                    fe_count += 1
                    if player_str:
                        stats.fe_by_player[player_str] = stats.fe_by_player.get(player_str, 0) + 1
                        fe_by_player_in_game[player_str] = fe_by_player_in_game.get(player_str, 0) + 1

                # Track recovery slide
                if move_type == "recovery_slide":
                    has_recovery_slide = True
                    recovery_mode = m.get("recoveryMode") or m.get("recovery_mode")
                    if isinstance(recovery_mode, str) and recovery_mode:
                        stats.recovery_slides_by_mode[recovery_mode] = (
                            stats.recovery_slides_by_mode.get(recovery_mode, 0) + 1
                        )
                        if recovery_mode == "stack_strike":
                            has_stack_strike = True
                            if player_str:
                                stack_strikes_by_player_in_game.add(player_str)
                    if player_str:
                        recovery_slides_by_player_in_game[player_str] = (
                            recovery_slides_by_player_in_game.get(player_str, 0) + 1
                        )
                        stats.recovery_slides_by_player[player_str] = (
                            stats.recovery_slides_by_player.get(player_str, 0) + 1
                        )

                # Track capture chains
                if move_type == "overtaking_capture":
                    stats.total_captures += 1
                    # Track first capture
                    if first_capture_player is None and player_str:
                        first_capture_player = player_str
                        stats.first_capture_by_player[player_str] = (
                            stats.first_capture_by_player.get(player_str, 0) + 1
                        )
                    if in_chain:
                        capture_chain_length += 1
                    else:
                        in_chain = True
                        capture_chain_length = 1
                elif move_type == "continue_capture_segment":
                    stats.total_chain_captures += 1
                    capture_chain_length += 1
                else:
                    # Chain ended
                    if capture_chain_length > max_chain_in_game:
                        max_chain_in_game = capture_chain_length
                    capture_chain_length = 0
                    in_chain = False

            # Final chain check
            if capture_chain_length > max_chain_in_game:
                max_chain_in_game = capture_chain_length

            # Record active moves
            stats.active_moves_per_game.append(active_move_count)

            # Record max skip streak for this game
            if max_skip_streak > 0:
                stats.longest_skip_streak_per_game.append(max_skip_streak)

            if has_fe:
                stats.games_with_fe += 1
                stats.fe_counts_per_game.append(fe_count)
                # Check if winner had forced elimination (comeback)
                if winner is not None and str(winner) in fe_by_player_in_game:
                    stats.games_with_late_fe_winner += 1

            if has_recovery_slide:
                stats.games_with_recovery_slide += 1
                # Check if winner used recovery slide
                if winner is not None and str(winner) in recovery_slides_by_player_in_game:
                    stats.wins_with_recovery_slide += 1
                elif winner is not None:
                    stats.wins_without_recovery_slide += 1
            elif winner is not None:
                stats.wins_without_recovery_slide += 1

            if has_stack_strike:
                stats.games_with_stack_strike += 1
                if winner is not None and str(winner) in stack_strikes_by_player_in_game:
                    stats.wins_with_stack_strike += 1

            if max_chain_in_game > 0:
                stats.max_capture_chain_lengths.append(max_chain_in_game)

            # First capture correlation with winning
            if first_capture_player is not None and winner is not None:
                if first_capture_player == str(winner):
                    stats.first_capturer_wins += 1
                else:
                    stats.first_capturer_loses += 1

        if any_games:
            report.data_sources.append(str(jsonl_path))


def collect_stats(
    data_dir: Path,
    jsonl_files: list[Path] | None = None,
    *,
    include_winner_only: bool = False,
    game_cutoff_ts: float | None = None,
    include_unknown_game_timestamp: bool = False,
    quarantine_writer: QuarantineWriter | None = None,
    data_quality: DataQualityMetrics | None = None,
    exclude_quarantined: bool = True,
    ai_types_filter: set[str] | None = None,
) -> AnalysisReport:
    """Collect statistics from all subdirectories in data_dir and optional JSONL files."""
    report = AnalysisReport()
    if data_quality is not None:
        report.data_quality = data_quality

    # Process JSONL files first if provided
    if jsonl_files:
        collect_stats_from_jsonl(
            jsonl_files,
            report,
            include_winner_only=include_winner_only,
            game_cutoff_ts=game_cutoff_ts,
            include_unknown_game_timestamp=include_unknown_game_timestamp,
            quarantine_writer=quarantine_writer,
            data_quality=data_quality,
            exclude_quarantined=exclude_quarantined,
            ai_types_filter=ai_types_filter,
        )

    if data_dir.exists():
        # Find all stats.json files
        for stats_path in data_dir.rglob("stats.json"):
            data = load_stats_json(stats_path)
            if data is None:
                continue

            board_type = data.get("board_type", "unknown")
            num_players = data.get("num_players", 2)
            key = (board_type, num_players)

            if key not in report.stats_by_config:
                report.stats_by_config[key] = GameStats(board_type=board_type, num_players=num_players)

            stats = report.stats_by_config[key]
            stats.total_games += data.get("total_games", 0)
            stats.total_moves += data.get("total_moves", 0)
            stats.total_time_seconds += data.get("total_time_seconds", 0.0)
            stats.draws += data.get("draws", 0)
            stats.games_with_recovery += data.get("games_with_recovery_opportunities", 0)
            stats.total_recovery_opportunities += data.get("total_recovery_opportunities", 0)
            stats.games_with_fe += data.get("games_with_fe", 0)

            # Merge wins by player
            for player, wins in data.get("wins_by_player", {}).items():
                stats.wins_by_player[player] = stats.wins_by_player.get(player, 0) + wins

            # Merge victory types
            victory_types = data.get("victory_type_counts", data.get("victory_types", {}))
            for vtype, count in victory_types.items():
                stats.victory_types[vtype] = stats.victory_types.get(vtype, 0) + count

            # Merge stalemate tiebreaker breakdown
            stalemate_tiebreakers = data.get("stalemate_by_tiebreaker", {})
            for tiebreaker, count in stalemate_tiebreakers.items():
                stats.stalemate_by_tiebreaker[tiebreaker] = (
                    stats.stalemate_by_tiebreaker.get(tiebreaker, 0) + count
                )

            # Collect individual game lengths if available
            game_lengths = data.get("game_lengths", [])
            if game_lengths:
                stats.game_lengths.extend(game_lengths)

            report.data_sources.append(str(stats_path))

        # Load recovery analysis if available
        recovery_data = load_recovery_analysis(data_dir)
        if recovery_data:
            report.recovery_analysis = recovery_data

    return report


def generate_markdown_report(report: AnalysisReport) -> str:
    """Generate a markdown report from the analysis."""
    lines = []
    lines.append("# RingRift Game Statistics Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Total Games Analyzed:** {report.total_games():,}")
    lines.append(f"**Total Moves:** {report.total_moves():,}")
    lines.append(f"**Data Sources:** {len(report.data_sources)}")
    lines.append("")

    # Aggregated Victory Categories
    lines.append("## 1. Aggregated Victory Categories")
    lines.append("")
    lines.append("*Categories: Territory (territory + stalemate-territory), ")
    lines.append("Elimination (elimination + ring_elimination + stalemate-ring_elim), LPS (last player standing)*")
    lines.append("")
    lines.append("| Board/Players | Games | Territory | Elimination | LPS |")
    lines.append("|---------------|-------|-----------|-------------|-----|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        agg = stats.get_aggregated_victory_counts()
        territory = agg["territory"]
        elimination = agg["elimination"]
        lps = agg["lps"]

        territory_pct = f"{100 * territory / stats.total_games:.1f}%" if territory else "0%"
        elimination_pct = f"{100 * elimination / stats.total_games:.1f}%" if elimination else "0%"
        lps_pct = f"{100 * lps / stats.total_games:.1f}%" if lps else "0%"

        lines.append(
            f"| {board_type} {num_players}p | {stats.total_games} | "
            f"{territory_pct} ({territory}) | {elimination_pct} ({elimination}) | {lps_pct} ({lps}) |"
        )

    lines.append("")

    # Detailed Victory Type Distribution (raw)
    lines.append("## 2. Detailed Victory Type Distribution")
    lines.append("")
    lines.append("| Board/Players  | Games | Territory   | Elimination | LPS        | Stalemate   | Timeout    |")
    lines.append("|----------------|-------|-------------|-------------|------------|-------------|------------|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        territory = stats.victory_types.get("territory", 0)
        # "elimination" and "ring_elimination" may both exist depending on normalization
        elimination = stats.victory_types.get("elimination", 0) + stats.victory_types.get("ring_elimination", 0)
        lps = stats.victory_types.get("lps", 0)
        stalemate = stats.victory_types.get("stalemate", 0)
        timeout = stats.victory_types.get("timeout", 0)

        def fmt_pct(count: int, width: int = 11) -> str:
            if count == 0:
                return "-".center(width)
            val = f"{count} ({100 * count / stats.total_games:.0f}%)"
            return val.ljust(width)

        board_col = f"{board_type} {num_players}p".ljust(14)
        games_col = str(stats.total_games).rjust(5)
        lines.append(
            f"| {board_col} | {games_col} | "
            f"{fmt_pct(territory)} | {fmt_pct(elimination)} | "
            f"{fmt_pct(lps, 10)} | {fmt_pct(stalemate)} | {fmt_pct(timeout, 10)} |"
        )

    lines.append("")

    # Stalemate Tiebreaker Breakdown
    any_stalemate_data = any(
        stats.stalemate_by_tiebreaker for stats in report.stats_by_config.values()
    )
    if any_stalemate_data:
        lines.append("### Stalemate Tiebreaker Breakdown")
        lines.append("")
        lines.append("| Board/Players | Total Stalemate | Territory | Ring Elim | Other |")
        lines.append("|---------------|-----------------|-----------|-----------|-------|")

        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            total_stalemate = stats.victory_types.get("stalemate", 0)
            if total_stalemate == 0:
                continue
            territory_tb = stats.stalemate_by_tiebreaker.get("territory", 0)
            ring_elim_tb = stats.stalemate_by_tiebreaker.get("ring_elimination", 0)
            other_tb = total_stalemate - territory_tb - ring_elim_tb

            lines.append(
                f"| {board_type} {num_players}p | {total_stalemate} | "
                f"{territory_tb} | {ring_elim_tb} | {other_tb} |"
            )

        lines.append("")

    # Win Distribution by Player Position
    lines.append("## 2. Win Distribution by Player Position")
    lines.append("")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        lines.append(f"### {board_type.title()} {num_players}-player ({stats.total_games} games)")
        lines.append("")
        lines.append("| Player | Wins | Win Rate |")
        lines.append("|--------|------|----------|")

        for p in range(1, num_players + 1):
            wins = stats.wins_by_player.get(str(p), 0)
            rate = 100 * stats.win_rate(p)
            lines.append(f"| Player {p} | {wins} | {rate:.1f}% |")

        # Calculate expected win rate and deviation
        expected = 100 / num_players
        lines.append("")
        lines.append(f"*Expected win rate (uniform): {expected:.1f}%*")
        lines.append("")

    # Game Length Statistics
    lines.append("## 3. Game Length Statistics")
    lines.append("")
    lines.append("| Configuration | Avg Moves | Min | Max | Median | Std Dev | Games/Sec | Total Time |")
    lines.append("|--------------|-----------|-----|-----|--------|---------|-----------|------------|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        avg_moves = stats.moves_per_game
        min_moves = stats.min_moves
        max_moves = stats.max_moves
        median_moves = stats.median_moves
        std_moves = stats.std_moves
        gps = stats.games_per_second
        total_time = stats.total_time_seconds

        time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time / 60:.1f}m"
        # Show detailed stats if game_lengths data is available
        if stats.game_lengths:
            lines.append(
                f"| {board_type} {num_players}p | {avg_moves:.1f} | {min_moves} | {max_moves} | "
                f"{median_moves:.1f} | {std_moves:.1f} | {gps:.3f} | {time_str} |"
            )
        else:
            lines.append(
                f"| {board_type} {num_players}p | {avg_moves:.1f} | - | - | "
                f"- | - | {gps:.3f} | {time_str} |"
            )

    lines.append("")

    # Rules/config drift detection (rings/thresholds).
    any_rings_data = any(
        stats.starting_rings_per_player_counts or stats.victory_threshold_counts
        for stats in report.stats_by_config.values()
    )
    if any_rings_data:
        lines.append("## 4. Rules/Config Sanity Checks")
        lines.append("")
        lines.append("*Detects mixed datasets (e.g., different rings-per-player or LPS thresholds) that can silently skew conclusions.*")
        lines.append("")
        lines.append("| Config | Starting Rings/Player (top) | Victory Threshold (top) | Territory Threshold (top) | LPS Rounds (top) |")
        lines.append("|--------|-----------------------------|-------------------------|---------------------------|-----------------|")

        def fmt_top_counts(counts: dict[str, int]) -> str:
            if not counts:
                return "-"
            items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]
            return ", ".join(f"{k}×{v}" for k, v in items)

        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            lines.append(
                f"| {board_type} {num_players}p | "
                f"{fmt_top_counts(stats.starting_rings_per_player_counts)} | "
                f"{fmt_top_counts(stats.victory_threshold_counts)} | "
                f"{fmt_top_counts(stats.territory_victory_threshold_counts)} | "
                f"{fmt_top_counts(stats.lps_rounds_required_counts)} |"
            )
        lines.append("")

    # Recovery Action Analysis
    if report.recovery_analysis:
        lines.append("## 5. Recovery Action Analysis")
        lines.append("")
        ra = report.recovery_analysis

        total = ra.get("total_games", 0)
        with_fe = ra.get("games_with_fe", 0)
        with_recovery = ra.get("games_with_recovery", 0)
        states = ra.get("states_analyzed", 0)

        lines.append(f"- **Games Analyzed:** {total}")
        lines.append(f"- **Games with Forced Elimination:** {with_fe} ({100 * with_fe / total:.1f}%)" if total else "")
        lines.append(f"- **Games with Recovery Used:** {with_recovery}")
        lines.append(f"- **States Analyzed:** {states:,}")
        lines.append("")

        # Condition frequencies
        cond_freq = ra.get("condition_frequencies", {})
        if cond_freq:
            lines.append("### Recovery Condition Frequencies")
            lines.append("")
            lines.append("| Condition | Frequency | % of States |")
            lines.append("|-----------|-----------|-------------|")
            for cond, count in sorted(cond_freq.items(), key=lambda x: -x[1]):
                pct = 100 * count / states if states else 0
                lines.append(f"| {cond} | {count:,} | {pct:.1f}% |")
            lines.append("")

        # Conditions met distribution
        cond_dist = ra.get("conditions_met_distribution", {})
        if cond_dist:
            lines.append("### Conditions Met Distribution")
            lines.append("")
            lines.append("| # Conditions | States | % |")
            lines.append("|--------------|--------|---|")
            for key, count in sorted(cond_dist.items()):
                pct = 100 * count / states if states else 0
                lines.append(f"| {key} | {count:,} | {pct:.1f}% |")
            lines.append("")

    # Enhanced Move Analysis Section
    any_move_data = any(stats.move_type_counts for stats in report.stats_by_config.values())
    if any_move_data:
        lines.append("## 6. Move Type Distribution")
        lines.append("")
        lines.append("*Top 10 move types across all configurations*")
        lines.append("")
        # Aggregate move types
        all_move_types: dict[str, int] = {}
        for stats in report.stats_by_config.values():
            for mtype, count in stats.move_type_counts.items():
                all_move_types[mtype] = all_move_types.get(mtype, 0) + count
        total_moves = sum(all_move_types.values())
        lines.append("| Move Type | Count | % of All Moves |")
        lines.append("|-----------|-------|----------------|")
        for mtype, count in sorted(all_move_types.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / total_moves if total_moves else 0
            lines.append(f"| {mtype} | {count:,} | {pct:.1f}% |")
        lines.append("")

    # Recovery Slide Analysis
    any_recovery_data = any(stats.games_with_recovery_slide > 0 for stats in report.stats_by_config.values())
    lines.append("## 7. Recovery Slide Analysis")
    lines.append("")
    if any_recovery_data:
        lines.append("| Config | Games w/ Recovery | Recovery User Wins | Non-Recovery Wins (in recovery games) |")
        lines.append("|--------|-------------------|--------------------|-----------------------------------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            games_w = stats.games_with_recovery_slide
            wins_w = stats.wins_with_recovery_slide
            # In games where recovery was used, how many were won by non-recovery user
            # This is tracked in wins_without_recovery_slide only for games WITH recovery
            # But our current tracking conflates "games with no recovery" and "recovery games where winner didn't use recovery"
            # Let's show: games_w, wins_w (recovery user won), and games_w - wins_w (non-recovery winner in recovery games)
            pct_games = 100 * games_w / stats.total_games if stats.total_games else 0
            # Out of games with recovery, what % were won by recovery user
            pct_recovery_wins = 100 * wins_w / games_w if games_w else 0
            # Non-recovery wins in recovery games (approximate: games_w had games_w games with winners)
            non_recovery_wins_in_recovery_games = games_w - wins_w if games_w > 0 else 0
            pct_non_recovery = 100 * non_recovery_wins_in_recovery_games / games_w if games_w else 0
            lines.append(
                f"| {board_type} {num_players}p | {games_w} ({pct_games:.1f}%) | "
                f"{wins_w} ({pct_recovery_wins:.1f}%) | {non_recovery_wins_in_recovery_games} ({pct_non_recovery:.1f}%) |"
            )
        lines.append("")
    else:
        lines.append("*No recovery slides detected in the analyzed games.*")
        lines.append("")

    any_recovery_mode_data = any(stats.recovery_slides_by_mode for stats in report.stats_by_config.values())
    if any_recovery_mode_data:
        lines.append("### Recovery Mode Breakdown")
        lines.append("")
        lines.append("| Config | Line | Fallback | Stack-Strike | Games w/ Stack-Strike | Winner Used Stack-Strike |")
        lines.append("|--------|------|----------|--------------|------------------------|--------------------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            line_count = stats.recovery_slides_by_mode.get("line", 0)
            fallback_count = stats.recovery_slides_by_mode.get("fallback", 0)
            strike_count = stats.recovery_slides_by_mode.get("stack_strike", 0)
            lines.append(
                f"| {board_type} {num_players}p | {line_count} | {fallback_count} | {strike_count} | "
                f"{stats.games_with_stack_strike} | {stats.wins_with_stack_strike} |"
            )
        lines.append("")

    # Capture Chain Analysis
    any_capture_data = any(stats.max_capture_chain_lengths for stats in report.stats_by_config.values())
    if any_capture_data:
        lines.append("## 8. Capture Chain Analysis")
        lines.append("")
        lines.append("| Config | Total Captures | Chain Captures | Max Chain | Avg Max Chain |")
        lines.append("|--------|----------------|----------------|-----------|---------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            total_cap = stats.total_captures
            chain_cap = stats.total_chain_captures
            max_chain = max(stats.max_capture_chain_lengths) if stats.max_capture_chain_lengths else 0
            avg_max = sum(stats.max_capture_chain_lengths) / len(stats.max_capture_chain_lengths) if stats.max_capture_chain_lengths else 0
            lines.append(f"| {board_type} {num_players}p | {total_cap:,} | {chain_cap:,} | {max_chain} | {avg_max:.1f} |")
        lines.append("")

    # Forced Elimination Analysis
    any_fe_data = any(stats.games_with_fe > 0 for stats in report.stats_by_config.values())
    if any_fe_data:
        lines.append("## 9. Forced Elimination Analysis")
        lines.append("")
        lines.append("| Config | Games w/ FE | FE Count | Winner Had FE | Comeback Rate |")
        lines.append("|--------|-------------|----------|---------------|---------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            games_fe = stats.games_with_fe
            pct_fe = 100 * games_fe / stats.total_games if stats.total_games else 0
            total_fe = sum(stats.fe_counts_per_game)
            comebacks = stats.games_with_late_fe_winner
            comeback_rate = 100 * comebacks / games_fe if games_fe else 0
            lines.append(
                f"| {board_type} {num_players}p | {games_fe} ({pct_fe:.1f}%) | "
                f"{total_fe} | {comebacks} | {comeback_rate:.1f}% |"
            )
        lines.append("")

    # First Blood Analysis
    any_first_capture = any(stats.first_capture_by_player for stats in report.stats_by_config.values())
    if any_first_capture:
        lines.append("## 10. First Blood Analysis")
        lines.append("")
        lines.append("*Does making the first capture correlate with winning?*")
        lines.append("")
        lines.append("| Config | First Capturer Wins | First Capturer Loses | Win Rate |")
        lines.append("|--------|---------------------|----------------------|----------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0 or not stats.first_capture_by_player:
                continue
            fc_wins = stats.first_capturer_wins
            fc_loses = stats.first_capturer_loses
            total_fc = fc_wins + fc_loses
            win_rate = 100 * fc_wins / total_fc if total_fc else 0
            expected = 100 / num_players
            delta = win_rate - expected
            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
            lines.append(
                f"| {board_type} {num_players}p | {fc_wins} | {fc_loses} | "
                f"{win_rate:.1f}% ({delta_str}% vs expected) |"
            )
        lines.append("")

    # Game Tempo/Activity Analysis
    any_activity_data = any(stats.active_moves_per_game for stats in report.stats_by_config.values())
    if any_activity_data:
        lines.append("## 11. Game Activity Analysis")
        lines.append("")
        lines.append("*Active moves (excluding skip/no-action moves)*")
        lines.append("")
        lines.append("| Config | Avg Total Moves | Avg Active Moves | Activity Rate |")
        lines.append("|--------|-----------------|------------------|---------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0 or not stats.active_moves_per_game:
                continue
            avg_total = stats.moves_per_game
            avg_active = sum(stats.active_moves_per_game) / len(stats.active_moves_per_game)
            activity_rate = 100 * avg_active / avg_total if avg_total else 0
            lines.append(f"| {board_type} {num_players}p | {avg_total:.1f} | {avg_active:.1f} | {activity_rate:.1f}% |")
        lines.append("")

    any_timeout_diag = any(stats.timeout_move_count_hist for stats in report.stats_by_config.values())
    if any_timeout_diag:
        lines.append("## 12. Timeout Diagnostics")
        lines.append("")
        lines.append("*Flags suspiciously consistent timeouts (often a runner/config pathology rather than real gameplay).*")
        lines.append("")
        lines.append("| Config | Timeout Games | Unique Move Counts | Most Common Move Count |")
        lines.append("|--------|--------------:|------------------:|------------------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            timeout_total = stats.victory_types.get("timeout", 0)
            if timeout_total == 0:
                continue
            hist = stats.timeout_move_count_hist
            unique = len(hist)
            most_common = "-"
            if hist:
                k, v = sorted(hist.items(), key=lambda x: (-x[1], x[0]))[0]
                most_common = f"{k} ({v})"
            lines.append(f"| {board_type} {num_players}p | {timeout_total} | {unique} | {most_common} |")
        lines.append("")

    # Key Findings
    lines.append("## 13. Key Findings")
    lines.append("")

    # Position advantage analysis
    position_advantages = []
    for (board_type, num_players), stats in report.stats_by_config.items():
        if stats.total_games < 10:
            continue
        expected = 1.0 / num_players
        max_player = None
        max_rate = 0.0
        for p in range(1, num_players + 1):
            rate = stats.win_rate(p)
            if rate > max_rate:
                max_rate = rate
                max_player = p
        if max_player and max_rate > expected + 0.05:
            position_advantages.append((board_type, num_players, max_player, max_rate))

    if position_advantages:
        lines.append("### Position Advantages")
        for board, players, player, rate in position_advantages:
            lines.append(f"- **{board} {players}p:** Player {player} has advantage ({rate * 100:.1f}% win rate)")
        lines.append("")

    # Victory type patterns
    lines.append("### Victory Type Patterns")
    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games < 10:
            continue
        dominant = max(stats.victory_types.items(), key=lambda x: x[1], default=(None, 0))
        if dominant[0] and dominant[1] > 0:
            rate = 100 * dominant[1] / stats.total_games
            lines.append(f"- **{board_type} {num_players}p:** {dominant[0]} dominant ({rate:.0f}%)")
    lines.append("")

    # Recovery status
    if report.recovery_analysis:
        with_recovery = report.recovery_analysis.get("games_with_recovery", 0)
        dist = report.recovery_analysis.get("conditions_met_distribution", {}) or {}
        # Canonical RR-CANON-R110 eligibility is 3 conditions. Keep backward
        # compatibility with older analysis artifacts that used a (wrong) 4th
        # "zero rings in hand" condition.
        conditions_met = dist.get("3_conditions", dist.get("4_conditions", 0))
        lines.append("### Recovery Mechanic Status")
        if with_recovery == 0 and conditions_met > 0:
            lines.append(
                f"- **Recovery is NOT being used** despite {conditions_met} states meeting all eligibility conditions"
            )
            lines.append("- This suggests turn-skipping prevents recovery-eligible players from taking turns")
        elif with_recovery > 0:
            lines.append(f"- Recovery was used in {with_recovery} games")
        lines.append("")

    # AI Type Distribution
    if report.games_by_ai_type:
        lines.append("## 14. AI Type Distribution")
        lines.append("")
        lines.append("*Games grouped by inferred AI opponent type*")
        lines.append("")
        lines.append("| AI Type | Games | % of Total |")
        lines.append("|---------|-------|------------|")
        total = sum(report.games_by_ai_type.values())
        for ai_type, count in sorted(report.games_by_ai_type.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total else 0
            lines.append(f"| {ai_type} | {count:,} | {pct:.1f}% |")
        lines.append("")

        # Per-config AI type breakdown (if multiple AI types exist)
        if len(report.games_by_ai_type) > 1:
            lines.append("### AI Type by Configuration")
            lines.append("")
            lines.append("| Config | " + " | ".join(sorted(report.games_by_ai_type.keys())) + " |")
            lines.append("|--------| " + " | ".join(["---"] * len(report.games_by_ai_type)) + " |")
            for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
                if stats.total_games == 0:
                    continue
                row = f"| {board_type} {num_players}p"
                for ai_type in sorted(report.games_by_ai_type.keys()):
                    count = stats.games_by_ai_type.get(ai_type, 0)
                    row += f" | {count}"
                row += " |"
                lines.append(row)
            lines.append("")

    # Data Quality Report
    dq = report.data_quality
    if dq.total_games_processed > 0:
        lines.append("## 15. Data Quality Report")
        lines.append("")
        lines.append(f"**Total Records Processed:** {dq.total_games_processed:,}")
        lines.append("")

        # Metadata inference rates
        lines.append("### Metadata Quality")
        lines.append("")
        lines.append("| Metric | Explicit | Inferred | Inference Rate |")
        lines.append("|--------|----------|----------|----------------|")

        explicit_bt = dq.games_with_explicit_board_type
        inferred_bt = dq.games_with_inferred_board_type
        bt_rate = 100 * inferred_bt / dq.total_games_processed if dq.total_games_processed else 0
        lines.append(f"| Board Type | {explicit_bt:,} | {inferred_bt:,} | {bt_rate:.1f}% |")

        explicit_np = dq.games_with_explicit_num_players
        inferred_np = dq.games_with_inferred_num_players
        np_rate = 100 * inferred_np / dq.total_games_processed if dq.total_games_processed else 0
        lines.append(f"| Num Players | {explicit_np:,} | {inferred_np:,} | {np_rate:.1f}% |")

        explicit_vt = dq.games_with_explicit_victory_type
        inferred_vt = dq.games_with_inferred_victory_type
        vt_rate = 100 * inferred_vt / dq.total_games_processed if dq.total_games_processed else 0
        lines.append(f"| Victory Type | {explicit_vt:,} | {inferred_vt:,} | {vt_rate:.1f}% |")
        lines.append("")

        # Completeness
        lines.append("### Record Completeness")
        lines.append("")
        winner_rate = 100 * dq.games_with_winner / dq.total_games_processed if dq.total_games_processed else 0
        moves_rate = 100 * dq.games_with_moves / dq.total_games_processed if dq.total_games_processed else 0
        lines.append(f"- **With Winner:** {dq.games_with_winner:,} ({winner_rate:.1f}%)")
        lines.append(f"- **With Moves:** {dq.games_with_moves:,} ({moves_rate:.1f}%)")
        lines.append(f"- **Malformed Records:** {dq.malformed_records:,}")
        lines.append("")

        # Quarantine summary
        if dq.quarantined_by_reason:
            lines.append("### Quarantined Records")
            lines.append("")
            lines.append("| Reason | Count | % of Total |")
            lines.append("|--------|-------|------------|")
            total_quarantined = sum(dq.quarantined_by_reason.values())
            for reason, count in sorted(dq.quarantined_by_reason.items(), key=lambda x: -x[1]):
                pct = 100 * count / dq.total_games_processed if dq.total_games_processed else 0
                lines.append(f"| {reason} | {count:,} | {pct:.1f}% |")
            total_pct = 100 * total_quarantined / dq.total_games_processed if dq.total_games_processed else 0
            lines.append(f"| **Total Quarantined** | **{total_quarantined:,}** | **{total_pct:.1f}%** |")
            lines.append("")

        # Source host distribution
        if dq.games_by_source_host:
            lines.append("### Data Sources by Host")
            lines.append("")
            lines.append("| Host | Games | % of Total |")
            lines.append("|------|-------|------------|")
            for host, count in sorted(dq.games_by_source_host.items(), key=lambda x: -x[1]):
                pct = 100 * count / dq.total_games_processed if dq.total_games_processed else 0
                lines.append(f"| {host} | {count:,} | {pct:.1f}% |")
            lines.append("")

    return "\n".join(lines)


def generate_json_report(report: AnalysisReport) -> dict[str, Any]:
    """Generate a JSON-serializable report."""
    result: dict[str, Any] = {
        "timestamp": report.timestamp,
        "summary": {
            "total_games": report.total_games(),
            "total_moves": report.total_moves(),
            "data_sources": len(report.data_sources),
        },
        "configurations": {},
        "recovery_analysis": report.recovery_analysis,
        "games_by_ai_type": report.games_by_ai_type,
        "data_quality": {
            "total_games_processed": report.data_quality.total_games_processed,
            "games_with_explicit_board_type": report.data_quality.games_with_explicit_board_type,
            "games_with_inferred_board_type": report.data_quality.games_with_inferred_board_type,
            "games_with_explicit_num_players": report.data_quality.games_with_explicit_num_players,
            "games_with_inferred_num_players": report.data_quality.games_with_inferred_num_players,
            "games_with_explicit_victory_type": report.data_quality.games_with_explicit_victory_type,
            "games_with_inferred_victory_type": report.data_quality.games_with_inferred_victory_type,
            "games_with_winner": report.data_quality.games_with_winner,
            "games_without_winner": report.data_quality.games_without_winner,
            "games_with_moves": report.data_quality.games_with_moves,
            "games_without_moves": report.data_quality.games_without_moves,
            "malformed_records": report.data_quality.malformed_records,
            "quarantined_by_reason": report.data_quality.quarantined_by_reason,
            "games_by_source_host": report.data_quality.games_by_source_host,
        },
    }

    for (board_type, num_players), stats in report.stats_by_config.items():
        key = f"{board_type}_{num_players}p"
        config_data: dict[str, Any] = {
            "board_type": board_type,
            "num_players": num_players,
            "total_games": stats.total_games,
            "total_moves": stats.total_moves,
            "moves_per_game": stats.moves_per_game,
            "total_time_seconds": stats.total_time_seconds,
            "games_per_second": stats.games_per_second,
            "wins_by_player": stats.wins_by_player,
            "win_rates": {str(p): stats.win_rate(p) for p in range(1, num_players + 1)},
            "victory_types": stats.victory_types,
            "victory_type_rates": {
                vtype: stats.victory_type_rate(vtype) for vtype in stats.victory_types
            },
            "stalemate_by_tiebreaker": stats.stalemate_by_tiebreaker,
            "aggregated_victory_types": stats.get_aggregated_victory_counts(),
            "draws": stats.draws,
            "games_with_recovery": stats.games_with_recovery,
            "games_with_fe": stats.games_with_fe,
            "games_with_recovery_slide": stats.games_with_recovery_slide,
            "recovery_slides_by_player": stats.recovery_slides_by_player,
            "wins_with_recovery_slide": stats.wins_with_recovery_slide,
            "wins_without_recovery_slide": stats.wins_without_recovery_slide,
            "games_with_late_fe_winner": stats.games_with_late_fe_winner,
            "fe_by_player": stats.fe_by_player,
            "move_type_counts": stats.move_type_counts,
            "total_captures": stats.total_captures,
            "total_chain_captures": stats.total_chain_captures,
            "ring_placement_moves": stats.ring_placement_moves,
            "territory_claims": stats.territory_claims,
            "line_formations": stats.line_formations,
            "first_capture_by_player": stats.first_capture_by_player,
            "first_capturer_wins": stats.first_capturer_wins,
            "first_capturer_loses": stats.first_capturer_loses,
            "starting_rings_per_player_counts": stats.starting_rings_per_player_counts,
            "victory_threshold_counts": stats.victory_threshold_counts,
            "territory_victory_threshold_counts": stats.territory_victory_threshold_counts,
            "lps_rounds_required_counts": stats.lps_rounds_required_counts,
            "starting_rings_per_player_breakdown": stats.starting_rings_per_player_breakdown,
            "victory_threshold_breakdown": stats.victory_threshold_breakdown,
            "games_with_starting_rings_mismatch": stats.games_with_starting_rings_mismatch,
            "games_with_victory_threshold_mismatch": stats.games_with_victory_threshold_mismatch,
            "recovery_slides_by_mode": stats.recovery_slides_by_mode,
            "games_with_stack_strike": stats.games_with_stack_strike,
            "wins_with_stack_strike": stats.wins_with_stack_strike,
            "timeout_move_count_hist": stats.timeout_move_count_hist,
            # AI type stats
            "games_by_ai_type": stats.games_by_ai_type,
            "wins_by_ai_type": stats.wins_by_ai_type,
            "victory_types_by_ai_type": stats.victory_types_by_ai_type,
            # Data quality stats
            "games_with_inferred_board_type": stats.games_with_inferred_board_type,
            "games_with_inferred_num_players": stats.games_with_inferred_num_players,
            "games_with_inferred_victory_type": stats.games_with_inferred_victory_type,
            "games_with_missing_winner": stats.games_with_missing_winner,
            "games_with_missing_moves": stats.games_with_missing_moves,
            "source_files_count": len(stats.source_files),
        }
        # Add detailed game length statistics if available
        if stats.game_lengths:
            config_data["game_length_stats"] = {
                "min": stats.min_moves,
                "max": stats.max_moves,
                "median": stats.median_moves,
                "std_dev": stats.std_moves,
            }
        if stats.active_moves_per_game:
            config_data["active_moves_stats"] = {
                "avg": sum(stats.active_moves_per_game) / len(stats.active_moves_per_game),
                "min": min(stats.active_moves_per_game),
                "max": max(stats.active_moves_per_game),
            }
        if stats.longest_skip_streak_per_game:
            config_data["skip_streak_stats"] = {
                "avg_longest": sum(stats.longest_skip_streak_per_game) / len(stats.longest_skip_streak_per_game),
                "max_longest": max(stats.longest_skip_streak_per_game),
            }
        if stats.max_capture_chain_lengths:
            config_data["capture_chain_stats"] = {
                "max_chain": max(stats.max_capture_chain_lengths),
                "avg_max_chain": sum(stats.max_capture_chain_lengths) / len(stats.max_capture_chain_lengths),
            }
        result["configurations"][key] = config_data

    return result


def _read_jsonl_filelist(filelist_path: Path) -> list[Path]:
    """Read a list of JSONL paths from a text/TSV file.

    Supports:
    - One path per line
    - TSV manifests with the path as the last column (e.g. mtime\\tsize\\tpath)
    """
    try:
        lines = filelist_path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        print(f"Warning: Failed to read jsonl file list {filelist_path}: {e}", file=sys.stderr)
        return []

    candidates: list[Path] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if not parts:
            continue
        raw = os.path.expanduser(parts[-1])
        path = Path(raw)
        if not path.is_absolute():
            local_candidate = filelist_path.parent / path
            if local_candidate.exists():
                path = local_candidate
            else:
                path = AI_SERVICE_ROOT / path
        candidates.append(path)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    result: list[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _expand_jsonl_globs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for p in paths:
        if p.exists():
            expanded.append(p)
            continue
        s = str(p)
        if any(ch in s for ch in ["*", "?", "["]):
            expanded.extend(Path(match) for match in glob.glob(s))
        else:
            expanded.append(p)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    result: list[Path] = []
    for p in expanded:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze RingRift selfplay game statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/selfplay"),
        help="Directory containing selfplay data (default: data/selfplay)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: stdout for markdown, auto-named for json)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no completed games are found, emit an empty report instead of exiting non-zero.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        help="JSONL files to include in analysis (can specify multiple or use glob patterns)",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        help="Directory to scan for JSONL files (*.jsonl)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively scan subdirectories when using --jsonl-dir",
    )
    parser.add_argument(
        "--jsonl-filelist",
        type=Path,
        help="Text/TSV file listing JSONL files to include (path in last column for TSV).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        help="Only include JSONL files modified within the last N hours.",
    )
    parser.add_argument(
        "--game-max-age-hours",
        type=float,
        default=None,
        help=(
            "Only include games whose per-game timestamp (top-level timestamp or first move timestamp) "
            "is within the last N hours. This is a per-record filter (more accurate than file mtime)."
        ),
    )
    parser.add_argument(
        "--include-unknown-game-timestamp",
        action="store_true",
        help="When --game-max-age-hours is set, include games with no parseable timestamp.",
    )
    parser.add_argument(
        "--include-winner-only",
        action="store_true",
        help="Include records that have a winner but no moves/termination fields (often non-game logs).",
    )
    parser.add_argument(
        "--ai-types",
        type=str,
        nargs="+",
        help="Only include games with these AI types (e.g., --ai-types neural_net nn-minimax nnue-guided).",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        help="Directory to write quarantined records (malformed, timeout, unknown_board, no_winner).",
    )
    parser.add_argument(
        "--fix-in-place",
        action="store_true",
        help="Fix missing metadata (board_type, num_players, victory_type) in JSONL files in-place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --fix-in-place, show what would be changed without actually modifying files.",
    )
    parser.add_argument(
        "--include-quarantined",
        action="store_true",
        help="Include quarantined records in statistics (default: exclude).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean JSONL files: fix metadata AND remove quarantinable records (timeout, unknown_board, malformed). Use with --quarantine-dir to save removed records.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Collect JSONL files from various sources
    jsonl_files: list[Path] = []
    if args.jsonl:
        jsonl_files.extend(args.jsonl)
    if args.jsonl_dir and args.jsonl_dir.exists():
        if args.recursive:
            jsonl_files.extend(args.jsonl_dir.rglob("*.jsonl"))
            jsonl_files.extend(args.jsonl_dir.rglob("*.jsonl.gz"))
        else:
            jsonl_files.extend(args.jsonl_dir.glob("*.jsonl"))
            jsonl_files.extend(args.jsonl_dir.glob("*.jsonl.gz"))
    if args.jsonl_filelist:
        jsonl_files.extend(_read_jsonl_filelist(args.jsonl_filelist))

    jsonl_files = _expand_jsonl_globs(jsonl_files)
    missing_files = [p for p in jsonl_files if not p.exists()]
    if missing_files and not args.quiet:
        for p in missing_files[:10]:
            print(f"Warning: JSONL file not found: {p}", file=sys.stderr)
        if len(missing_files) > 10:
            print(f"Warning: ... and {len(missing_files) - 10} more missing JSONL files", file=sys.stderr)
    jsonl_files = [p for p in jsonl_files if p.exists()]
    if args.max_age_hours is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - (args.max_age_hours * 3600)
        jsonl_files = [p for p in jsonl_files if p.stat().st_mtime >= cutoff]

    game_cutoff_ts: float | None = None
    if args.game_max_age_hours is not None:
        game_cutoff_ts = datetime.now(timezone.utc).timestamp() - (float(args.game_max_age_hours) * 3600)

    # Handle fix-in-place mode
    if args.fix_in_place:
        if not jsonl_files:
            print("Error: --fix-in-place requires JSONL files (use --jsonl, --jsonl-dir, or --jsonl-filelist)", file=sys.stderr)
            return 1
        total_records = 0
        total_modified = 0
        for jsonl_path in jsonl_files:
            records, modified = fix_jsonl_in_place(
                jsonl_path,
                dry_run=bool(args.dry_run),
                quiet=bool(args.quiet),
            )
            total_records += records
            total_modified += modified
        if not args.quiet:
            mode = "Would fix" if args.dry_run else "Fixed"
            print(f"{mode} {total_modified:,} records across {len(jsonl_files)} files ({total_records:,} total records)", file=sys.stderr)
        return 0

    # Handle clean mode (fix metadata + remove bad records)
    if getattr(args, "clean", False):
        if not jsonl_files:
            print("Error: --clean requires JSONL files (use --jsonl, --jsonl-dir, or --jsonl-filelist)", file=sys.stderr)
            return 1

        quarantine_path = args.quarantine_dir if args.quarantine_dir else None

        total_result = CleanResult()
        for i, jsonl_path in enumerate(jsonl_files, 1):
            if not args.quiet and i % 100 == 0:
                print(f"Cleaning file {i}/{len(jsonl_files)}...", file=sys.stderr)
            result = clean_jsonl_file(
                jsonl_path,
                quarantine_dir=quarantine_path,
                dry_run=bool(args.dry_run),
                quiet=bool(args.quiet),
            )
            total_result.total_records += result.total_records
            total_result.kept_records += result.kept_records
            total_result.fixed_records += result.fixed_records
            total_result.malformed_records += result.malformed_records
            for reason, count in result.quarantined_by_reason.items():
                total_result.quarantined_by_reason[reason] = total_result.quarantined_by_reason.get(reason, 0) + count

        if not args.quiet:
            mode = "Would clean" if args.dry_run else "Cleaned"
            q_total = sum(total_result.quarantined_by_reason.values())
            print(f"\n{mode} {len(jsonl_files)} files:", file=sys.stderr)
            print(f"  Total records: {total_result.total_records:,}", file=sys.stderr)
            print(f"  Kept: {total_result.kept_records:,}", file=sys.stderr)
            print(f"  Fixed metadata: {total_result.fixed_records:,}", file=sys.stderr)
            print(f"  Quarantined: {q_total:,}", file=sys.stderr)
            if total_result.quarantined_by_reason:
                for reason, count in sorted(total_result.quarantined_by_reason.items(), key=lambda x: -x[1]):
                    print(f"    - {reason}: {count:,}", file=sys.stderr)
        return 0

    # Create quarantine writer if requested
    quarantine_writer: QuarantineWriter | None = None
    if args.quarantine_dir:
        quarantine_writer = QuarantineWriter(args.quarantine_dir)

    # Create data quality metrics tracker
    data_quality = DataQualityMetrics()

    # Determine whether to exclude quarantined records from stats
    exclude_quarantined = not getattr(args, "include_quarantined", False)

    # Check if we have any data sources
    has_data_dir = args.data_dir.exists()
    has_jsonl = len(jsonl_files) > 0

    if not has_data_dir and not has_jsonl:
        if args.allow_empty:
            report = AnalysisReport()
        else:
            print(f"Error: No data sources found. Data directory does not exist: {args.data_dir}", file=sys.stderr)
            return 1
    else:
        if not args.quiet:
            if has_data_dir:
                print(f"Analyzing data in {args.data_dir}...", file=sys.stderr)
            if has_jsonl:
                print(f"Including {len(jsonl_files)} JSONL file(s)...", file=sys.stderr)

        # Convert AI types filter to set if provided
        ai_types_filter = set(args.ai_types) if args.ai_types else None

        report = collect_stats(
            args.data_dir,
            jsonl_files if has_jsonl else None,
            include_winner_only=bool(args.include_winner_only),
            game_cutoff_ts=game_cutoff_ts,
            include_unknown_game_timestamp=bool(args.include_unknown_game_timestamp),
            quarantine_writer=quarantine_writer,
            data_quality=data_quality,
            exclude_quarantined=exclude_quarantined,
            ai_types_filter=ai_types_filter,
        )

        if report.total_games() == 0 and not args.allow_empty:
            print("Error: No game data found", file=sys.stderr)
            return 1

        if not args.quiet:
            print(
                f"Found {report.total_games()} games across {len(report.stats_by_config)} configurations",
                file=sys.stderr,
            )

    # Show quarantine summary if applicable
    if quarantine_writer and not args.quiet:
        counts = quarantine_writer.get_counts()
        if counts:
            print(f"\nQuarantined records written to {args.quarantine_dir}:", file=sys.stderr)
            for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count:,}", file=sys.stderr)
            print(f"  Total: {sum(counts.values()):,}", file=sys.stderr)

    # Generate outputs
    if args.format in ("markdown", "both"):
        md_report = generate_markdown_report(report)
        if args.output and args.format == "markdown":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_report)
            if not args.quiet:
                print(f"Markdown report written to {args.output}", file=sys.stderr)
        elif args.format == "both" and args.output:
            md_path = args.output.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_report)
            if not args.quiet:
                print(f"Markdown report written to {md_path}", file=sys.stderr)
        else:
            print(md_report)

    if args.format in ("json", "both"):
        json_report = generate_json_report(report)
        if args.output:
            json_path = args.output if args.format == "json" else args.output.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_report, f, indent=2)
            if not args.quiet:
                print(f"JSON report written to {json_path}", file=sys.stderr)
        elif args.format == "json":
            print(json.dumps(json_report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
