#!/usr/bin/env python
"""
Export training samples from existing GameReplayDB replays.

This script walks one or more GameReplayDB SQLite files and converts completed
games into a neural-net training dataset in the same NPZ format used by
app.training.generate_data:

    - features: (N, C, H, W) float32
    - globals:  (N, G)       float32
    - values:   (N,)         float32   (from final ranking, per-state perspective)
    - policy_indices: (N,)   object    → np.ndarray[int32] of indices per sample
    - policy_values:  (N,)   object    → np.ndarray[float32] of probs per sample

Each sample corresponds to a (state_before_move, move_taken) pair drawn from
recorded games, with an outcome label derived from the final player ranking.

**Value Target Encoding (rank-aware for multiplayer):**
  - 2-player: winner=+1, loser=-1 (unchanged)
  - 3-player: 1st=+1, 2nd=0, 3rd=-1
  - 4-player: 1st=+1, 2nd=+0.33, 3rd=-0.33, 4th=-1

**Quality Filtering:**
  - Use --require-completed to only include games with normal termination
  - Use --min-moves N to exclude trivially short games
  - Use --max-moves N to exclude abnormally long games

Usage examples (from ai-service/):

    # Basic: export square8 2p samples (parallel by default, uses CPU_COUNT-1 workers)
    python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --output data/training/from_replays.square8_2p.npz

    # Quality-filtered export with explicit worker count
    python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_3p.db \\
        --board-type square8 \\
        --num-players 3 \\
        --workers 16 \\
        --require-completed \\
        --min-moves 20 \\
        --output data/training/from_replays.square8_3p.npz

    # Single-threaded mode (for debugging or when parallel causes issues)
    python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --single-threaded \\
        --output data/training/debug.npz

    # Incremental export with caching (skip if DBs unchanged)
    python scripts/export_replay_dataset.py \\
        --db data/games/consolidated.db \\
        --board-type square8 --num-players 2 \\
        --output data/training/square8_2p.npz \\
        --use-cache

    # Force re-export even with valid cache
    python scripts/export_replay_dataset.py \\
        --db data/games/consolidated.db \\
        --board-type square8 --num-players 2 \\
        --output data/training/square8_2p.npz \\
        --use-cache --force-export
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sqlite3


# Dec 28, 2025: Disk space reservation to prevent partial writes
# Dec 29, 2025: Increased from 500MB to 2GB for safety margin after corruption incidents
DISK_SPACE_SAFETY_MARGIN_MB = 2048  # Keep 2GB free after write (was 500MB)
# January 2026: Made compression ratio configurable via env var for easier tuning
# Typical observed ratios: float16 features ~0.25-0.35, int8 policy ~0.40-0.50
# Default 0.35 is conservative (overestimates size slightly for safety)
NPZ_COMPRESSION_RATIO = float(os.getenv("RINGRIFT_NPZ_COMPRESSION_RATIO", "0.35"))


def _estimate_npz_size(save_kwargs: dict) -> int:
    """Estimate the compressed NPZ file size in bytes.

    Uses NPZ_COMPRESSION_RATIO (configurable via RINGRIFT_NPZ_COMPRESSION_RATIO env var).
    Typical observed values:
    - float16 features: ~0.25-0.35 (high entropy, compresses well)
    - int8 policy targets: ~0.40-0.50 (sparse, less compressible)
    - Mixed data: ~0.30-0.40

    Args:
        save_kwargs: Dictionary of arrays to save

    Returns:
        Estimated file size in bytes
    """
    total_bytes = 0
    for key, arr in save_kwargs.items():
        if isinstance(arr, np.ndarray):
            total_bytes += arr.nbytes
    # Apply compression ratio estimate
    return int(total_bytes * NPZ_COMPRESSION_RATIO)


def _check_disk_space_for_export(output_path: str, save_kwargs: dict) -> tuple[bool, str]:
    """Check if there's enough disk space for NPZ export.

    Args:
        output_path: Path where NPZ will be written
        save_kwargs: Dictionary of arrays to save

    Returns:
        Tuple of (has_space, message)
    """
    try:
        # Get destination directory
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not output_dir:
            output_dir = os.getcwd()

        # Ensure directory exists for disk_usage check
        os.makedirs(output_dir, exist_ok=True)

        # Check available space
        usage = shutil.disk_usage(output_dir)
        available_mb = usage.free / (1024 * 1024)

        # Estimate required space
        estimated_bytes = _estimate_npz_size(save_kwargs)
        estimated_mb = estimated_bytes / (1024 * 1024)
        required_mb = estimated_mb + DISK_SPACE_SAFETY_MARGIN_MB

        if available_mb < required_mb:
            return False, (
                f"Insufficient disk space for export: "
                f"need {required_mb:.1f}MB (estimated {estimated_mb:.1f}MB + "
                f"{DISK_SPACE_SAFETY_MARGIN_MB}MB safety margin), "
                f"available {available_mb:.1f}MB at {output_dir}"
            )

        return True, f"Disk space OK: {available_mb:.1f}MB available, need ~{required_mb:.1f}MB"

    except OSError as e:
        # If we can't check, log warning but don't block
        return True, f"Could not check disk space: {e}"

logger = logging.getLogger(__name__)


# Database lock retry configuration (Dec 2025: configurable via env vars)
# January 2026: Increased default retries from 5 to 10 to better handle cluster sync
# which can take 1-2 minutes. With exponential backoff: 0.5+1+2+4+8+16+30+30+30+30 = ~150s max
DB_LOCK_MAX_RETRIES = int(os.getenv("RINGRIFT_DB_LOCK_MAX_RETRIES", "10"))
DB_LOCK_INITIAL_WAIT = float(os.getenv("RINGRIFT_DB_LOCK_INITIAL_WAIT", "0.5"))  # seconds
DB_LOCK_MAX_WAIT = float(os.getenv("RINGRIFT_DB_LOCK_MAX_WAIT", "30.0"))  # seconds


def _is_db_locked_error(e: Exception) -> bool:
    """Check if exception is a database lock error."""
    error_str = str(e).lower()
    return (
        "database is locked" in error_str or
        "database is locked" in error_str or
        "locked" in error_str and "database" in error_str or
        isinstance(e, sqlite3.OperationalError) and "locked" in str(e)
    )


def _open_db_with_retry(
    db_path: str,
    max_retries: int = DB_LOCK_MAX_RETRIES,
    initial_wait: float = DB_LOCK_INITIAL_WAIT,
) -> "GameReplayDB":
    """Open GameReplayDB with retry logic for database lock errors.

    Uses exponential backoff when the database is locked (e.g., during cluster sync).

    Dec 27, 2025: Added to prevent export failures during cluster data sync.

    Args:
        db_path: Path to the SQLite database
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds (doubles each retry)

    Returns:
        GameReplayDB instance

    Raises:
        Exception: If database cannot be opened after all retries
    """
    last_error: Exception | None = None
    wait_time = initial_wait

    for attempt in range(max_retries + 1):
        try:
            return GameReplayDB(db_path)
        except Exception as e:
            last_error = e
            if not _is_db_locked_error(e):
                # Not a lock error - don't retry
                raise

            if attempt < max_retries:
                # Calculate wait with exponential backoff, capped at max
                actual_wait = min(wait_time, DB_LOCK_MAX_WAIT)
                logger.warning(
                    f"Database locked, retrying in {actual_wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries}): {db_path}"
                )
                time.sleep(actual_wait)
                wait_time *= 2  # Exponential backoff

    # All retries exhausted
    logger.error(
        f"Failed to open database after {max_retries} retries: {db_path}"
    )
    raise last_error

from app.ai.neural_net import INVALID_MOVE_INDEX, NeuralNetAI, encode_move_for_board
from app.db import GameReplayDB
from app.models import AIConfig, BoardType, GameState, Move, Position
from app.training.canonical_sources import enforce_canonical_sources
from app.training.data_quality import embed_checksums_in_save_kwargs
from app.training.mcts_labeling import build_move_key

# Unified game discovery
try:
    from app.utils.game_discovery import GameDiscovery
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None

# Quality scoring for filtering (December 2025)
try:
    from app.quality.unified_quality import compute_game_quality_from_params
    HAS_QUALITY_SCORER = True
except ImportError:
    HAS_QUALITY_SCORER = False
    compute_game_quality_from_params = None

# Heuristic feature extraction for v5 training (December 2025)
# Fast mode (21 features): O(1) component scores
# Full mode (49 features): Linear weight decomposition for maximum strength
try:
    from app.training.fast_heuristic_features import (
        extract_heuristic_features,
        extract_full_heuristic_features,
        HEURISTIC_FEATURE_NAMES,
        NUM_HEURISTIC_FEATURES,
        NUM_HEURISTIC_FEATURES_FULL,
    )
    HAS_HEURISTIC_EXTRACTOR = True
    HAS_FULL_HEURISTIC_EXTRACTOR = True
except ImportError:
    HAS_HEURISTIC_EXTRACTOR = False
    HAS_FULL_HEURISTIC_EXTRACTOR = False
    NUM_HEURISTIC_FEATURES = 21  # Fallback (fast mode)
    NUM_HEURISTIC_FEATURES_FULL = 49  # Fallback (full mode)

# Quality-weighted sampling with freshness (December 2025)
try:
    from app.training.source_weighting import compute_combined_weights
    HAS_SOURCE_WEIGHTING = True
except ImportError:
    HAS_SOURCE_WEIGHTING = False
    compute_combined_weights = None


def _normalize_hex_board_size(board: "BoardState") -> "BoardState":
    """Normalize hex board size from legacy Convention A to Convention B.

    Legacy hexagonal games stored board.size=13 (radius+1), but the encoder
    expects board.size=25 (2*radius+1 = bounding box). This function creates
    a normalized board with the correct size for encoding.

    Note: This modifies only the size attribute. Position data in board.stacks
    should already be in cube coordinates [-12, 12].
    """
    from app.models import BoardState, BoardType

    if board.type != BoardType.HEXAGONAL:
        return board

    # Convention A: size = radius + 1 = 13
    # Convention B: size = 2*radius + 1 = 25
    if board.size == 13:
        # Create a new BoardState with corrected size
        return BoardState(
            type=board.type,
            size=25,  # Correct bounding box size
            stacks=board.stacks,
        )

    return board


def _normalize_hex_move_coords(move: Move, board_type: BoardType, board_size: int) -> Move:
    """Normalize move positions from canvas to cube coords for hex boards.

    Legacy GPU selfplay stored hexagonal positions in canvas coords [0, board_size).
    The encoder expects cube coords [-radius, radius]. This function detects and
    converts canvas coords to cube coords for hex boards.

    Detection heuristic: If any coord > radius, it's likely canvas coords.
    For unambiguous cases (coords in [0, radius]), we assume cube coords.
    """
    if board_type not in (BoardType.HEXAGONAL, BoardType.HEX8):
        return move

    radius = (board_size - 1) // 2

    def maybe_convert(pos: Position | None) -> Position | None:
        if pos is None:
            return None

        # Check if coords look like canvas (any coord > radius means canvas)
        x, y = pos.x, pos.y
        if x > radius or y > radius or x < -radius or y < -radius:
            # These are definitely canvas coords [0, board_size) - convert to cube
            cube_x = x - radius
            cube_y = y - radius
            cube_z = -cube_x - cube_y
            return Position(x=cube_x, y=cube_y, z=cube_z)
        else:
            # Ambiguous or already cube coords - assume cube (z might need fixing)
            if pos.z is None:
                z = -x - y
                return Position(x=x, y=y, z=z)
            return pos

    # Create new Move with normalized positions
    return Move(
        id=move.id,
        type=move.type,
        player=move.player,
        from_pos=maybe_convert(move.from_pos),
        to=maybe_convert(move.to),
        capture_target=move.capture_target,
        captured_stacks=move.captured_stacks,
        capture_chain=move.capture_chain,
        overtaken_rings=move.overtaken_rings,
        placed_on_stack=move.placed_on_stack,
        placement_count=move.placement_count,
        stack_moved=move.stack_moved,
        minimum_distance=move.minimum_distance,
        actual_distance=move.actual_distance,
        marker_left=move.marker_left,
        line_index=move.line_index,
        formed_lines=move.formed_lines,
        collapsed_markers=move.collapsed_markers,
        claimed_territory=move.claimed_territory,
        disconnected_regions=move.disconnected_regions,
        recovery_option=move.recovery_option,
        recovery_mode=move.recovery_mode,
        collapse_positions=move.collapse_positions,
        extraction_stacks=move.extraction_stacks,
        eliminated_rings=move.eliminated_rings,
        elimination_context=move.elimination_context,
        timestamp=move.timestamp,
        think_time=move.think_time,
        move_number=move.move_number,
        phase=move.phase,
    )


import re

# Regex patterns for parsing full repr move_probs keys (from descent_ai's str(m) format)
_RE_MOVE_TYPE = re.compile(r"type=<MoveType\.\w+:\s*'(\w+)'>")
_RE_FROM_POS = re.compile(r"from_pos=Position\(x=(-?\d+),\s*y=(-?\d+)")
_RE_TO_POS = re.compile(r"\bto=Position\(x=(-?\d+),\s*y=(-?\d+)")


def _normalize_db_move_key(key: str) -> str:
    """Normalize a move_probs DB key to build_move_key format for matching.

    Handles two formats:
    1. Short format (from build_move_key): "place_ring_-4,4" -> unchanged
    2. Full repr format (from str(m)): "id='simulated' type=<MoveType.PLACE_RING: 'place_ring'>
       ... to=Position(x=-4, y=4, z=0) ..." -> "place_ring_-4,4"
    """
    if not key.startswith("id="):
        return key  # Already in short format

    type_match = _RE_MOVE_TYPE.search(key)
    if not type_match:
        return key
    type_val = type_match.group(1)

    from_match = _RE_FROM_POS.search(key)
    to_match = _RE_TO_POS.search(key)

    result = type_val
    if from_match:
        result += f"_{from_match.group(1)},{from_match.group(2)}"
    if to_match:
        result += f"_{to_match.group(1)},{to_match.group(2)}"
    return result


from app.training.encoding import get_encoder_for_board_type
from app.training.export_cache import get_export_cache
from app.training.export_core import (
    compute_multi_player_values,
    encode_state_with_history,
    value_from_final_ranking,
    value_from_final_winner,
)
from scripts.lib.cli import BOARD_TYPE_MAP


def _enforce_canonical_db_policy(
    db_paths: list[str],
    output_path: str,
    *,
    allow_noncanonical: bool,
) -> None:
    """Refuse to label outputs as canonical when source DBs are non-canonical."""
    if allow_noncanonical:
        return

    if not os.path.basename(output_path).startswith("canonical_"):
        return

    noncanonical = [path for path in db_paths if not os.path.basename(path).startswith("canonical_")]
    if noncanonical:
        joined = ", ".join(noncanonical)
        raise SystemExit(
            "[export-replay-dataset] Refusing to export canonical_* dataset from non-canonical DB(s): "
            f"{joined}\n"
            "Use --allow-noncanonical to override, or rename the output to avoid canonical_ prefix."
        )


def _check_remote_data_sources(
    board_type: str, num_players: int, local_db_paths: list[str]
) -> None:
    """Check remote sources for additional game data (January 2026).

    This is informational only - it logs what data is available remotely
    so users know if they should run sync before export.

    Args:
        board_type: Board type (e.g., "hex8")
        num_players: Number of players
        local_db_paths: List of local database paths already discovered
    """
    config_key = f"{board_type}_{num_players}p"
    remote_sources: list[tuple[str, int]] = []

    # 1. Check OWC external drive for databases
    try:
        import subprocess

        owc_host = os.environ.get("OWC_HOST", "mac-studio")
        owc_path = os.environ.get("OWC_BASE_PATH", "/Volumes/RingRift-Data")

        # Quick SSH check for OWC databases
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                owc_host,
                f"find {owc_path}/games -name '*{config_key}*.db' 2>/dev/null | wc -l",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            owc_count = int(result.stdout.strip())
            if owc_count > 0:
                remote_sources.append((f"OWC ({owc_host})", owc_count))
    except Exception:
        pass  # OWC not available, skip silently

    # 2. Check S3 for archived databases
    try:
        import subprocess

        s3_bucket = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
        result = subprocess.run(
            [
                "aws",
                "s3",
                "ls",
                f"s3://{s3_bucket}/consolidated/games/",
                "--recursive",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            # Count matching files
            s3_count = sum(
                1 for line in result.stdout.splitlines() if config_key in line
            )
            if s3_count > 0:
                remote_sources.append(("S3", s3_count))
    except Exception:
        pass  # S3 not available, skip silently

    # 3. Check ClusterManifest for games on other nodes
    try:
        from app.distributed.cluster_manifest import get_cluster_manifest

        manifest = get_cluster_manifest()
        remote_games = manifest.get_game_count(config_key)
        if remote_games > 0:
            remote_sources.append(("P2P Cluster", remote_games))
    except Exception:
        pass  # Manifest not available, skip silently

    # Report findings
    if remote_sources:
        print(f"\n[REMOTE DATA] Additional data available for {config_key}:")
        for source, count in remote_sources:
            if "games" in source.lower() or "P2P" in source:
                print(f"  - {source}: ~{count:,} games")
            else:
                print(f"  - {source}: {count} database(s)")
        print(
            "  Tip: Run 'python scripts/master_loop.py' or sync daemons to fetch remote data.\n"
        )


def build_encoder(
    board_type: BoardType,
    encoder_version: str = "default",
    feature_version: int = 2,
) -> NeuralNetAI:
    """
    Construct a NeuralNetAI instance for feature and policy encoding.

    This uses a lightweight AIConfig and treats player_number=1 purely as a
    placeholder; we never call select_move(), only the encoding helpers.

    For hexagonal boards, encoder_version can be:
      - "default": Maps to "v3" (HexStateEncoderV3, 16 channels)
      - "v2": Use HexStateEncoder (10 channels) for HexNeuralNet_v2
      - "v3": Use HexStateEncoderV3 (16 channels) for HexNeuralNet_v3

    feature_version controls the global feature layout for encoders.

    IMPORTANT: Hex boards ALWAYS use specialized encoders to ensure consistent
    feature shapes across all games. The "default" option maps to v3.
    """
    # Prefer CPU by default to avoid accidental MPS/OMP issues; callers can
    # override via env (e.g. RINGRIFT_FORCE_CPU=0) if they want GPU.
    os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

    config = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        rngSeed=None,
        heuristic_profile_id=None,
        nn_model_id=None,
        heuristic_eval_mode=None,
        use_neural_net=True,
    )
    encoder = NeuralNetAI(player_number=1, config=config)
    encoder.feature_version = int(feature_version)
    # Ensure the encoder's board_size hint is consistent with the dataset.
    encoder.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEX8: 9,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)

    # For hex boards, ALWAYS attach a specialized encoder to ensure consistent
    # feature shapes. Default to v3 (newest, 16 channels).
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        effective_version = encoder_version if encoder_version in ("v2", "v3") else "v3"
        encoder._hex_encoder = get_encoder_for_board_type(
            board_type,
            effective_version,
            feature_version=feature_version,
        )
        encoder._hex_encoder_version = effective_version

    return encoder


# Value computation and encoding functions are now imported from app.training.export_core


def export_replay_dataset_multi(
    db_paths: list[str],
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    feature_version: int = 2,
    sample_every: int = 1,
    max_games: int | None = None,
    require_completed: bool = False,
    min_moves: int | None = None,
    max_moves: int | None = None,
    max_move_index: int | None = None,
    use_rank_aware_values: bool = True,
    parity_fixtures_dir: str | None = None,
    exclude_recovery: bool = False,
    use_board_aware_encoding: bool = True,  # Default to board-aware encoding (not legacy MAX_N)
    append: bool = False,
    encoder_version: str = "default",
    require_moves: bool = True,
    min_quality: float | None = None,  # December 2025: Quality filtering
    min_elo: float | None = None,  # January 2026: Elo-gated training (filter by generator Elo)
    include_heuristics: bool = False,  # December 2025: Extract heuristic features for v5
    full_heuristics: bool = False,  # December 2025: Use full 49-feature extraction
    # Source filtering (December 2025 - Phase 5 Unified NN/NNUE training)
    include_sources: set[str] | None = None,  # Source types to include (None=selfplay only)
    exclude_sources: set[str] | None = None,  # Source types to exclude
    # Move data validation (January 2026 - Phase 6 Data Integrity)
    fail_on_orphans: bool = True,  # Fail export if orphan games found (no move data)
    # Quality-weighted sampling (December 2025)
    quality_weighted: bool = False,  # Use quality-weighted sample weights
    # Quality tier filtering (January 2026 - Training Quality Fix)
    quality_tier: str | None = None,  # Filter by engine mode: bootstrap, standard, quality, ultimate
) -> None:
    """
    Export training samples from multiple GameReplayDB files into an NPZ dataset
    with automatic deduplication by game_id.

    This function processes databases in order, tracking game_ids to skip
    duplicates that appear in multiple sources. This enables aggregated training
    from siloed data across multiple nodes without double-counting games.

    Args:
        db_paths: List of paths to GameReplayDB SQLite files
        board_type: Board type to filter games by
        num_players: Number of players to filter games by
        output_path: Path to output .npz dataset
        history_length: Number of past feature frames to stack (default: 3)
        feature_version: Feature encoding version for global feature layout
        sample_every: Use every Nth move as a training sample (default: 1)
        max_games: Optional cap on total number of games to process across all DBs
        require_completed: If True, only include games with normal termination
        min_moves: Minimum move count to include a game
        max_moves: Maximum move count to include a game
        use_rank_aware_values: If True, use rank-based values for multiplayer
        use_board_aware_encoding: If True, use board-specific policy encoding
        append: If True, append to existing output NPZ
        encoder_version: Encoder version for hex boards ('default', 'v2', 'v3')
        require_moves: If True, only include games with move data (default: True)
        fail_on_orphans: If True (default), fail export if any database contains
            orphan games (games with total_moves > 0 but no game_moves records).
            This prevents corrupt training data. Set to False to warn and continue.

    Raises:
        MoveDataValidationError: If fail_on_orphans=True and orphan games are found.
    """
    encoder = build_encoder(
        board_type,
        encoder_version=encoder_version,
        feature_version=feature_version,
    )

    features_list: list[np.ndarray] = []
    globals_list: list[np.ndarray] = []
    values_list: list[float] = []
    values_mp_list: list[np.ndarray] = []
    num_players_list: list[int] = []
    policy_indices_list: list[np.ndarray] = []
    policy_values_list: list[np.ndarray] = []
    move_numbers_list: list[int] = []
    total_game_moves_list: list[int] = []
    phases_list: list[str] = []
    victory_types_list: list[str] = []  # For victory-type-balanced sampling
    engine_modes_list: list[str] = []  # For source-based sample weighting (Gumbel 3x weight)
    move_types_list: list[str] = []  # For chain-aware sample weighting
    opponent_elo_list: list[float] = []  # For ELO-weighted training (December 2025)
    generator_elo_list: list[float] = []  # For quality-weighted sampling by generator strength (January 2026)
    opponent_types_list: list[str] = []  # For opponent diversity tracking (December 2025)
    quality_score_list: list[float] = []  # For quality-weighted training (December 2025)
    heuristics_list: list[np.ndarray] = []  # For v5 heavy training (December 2025)
    timestamps_list: list[float] = []  # For freshness-weighted sampling (December 2025)

    # Validate heuristic extraction availability
    if include_heuristics and not HAS_HEURISTIC_EXTRACTOR:
        print("Warning: --include-heuristics requested but heuristic extractor not available")
        include_heuristics = False
    if full_heuristics and not HAS_FULL_HEURISTIC_EXTRACTOR:
        print("Warning: --full-heuristics requested but full extractor not available")
        full_heuristics = False
    if full_heuristics and not include_heuristics:
        print("Note: --full-heuristics implies --include-heuristics")
        include_heuristics = True

    # Determine the heuristic feature count for this export
    effective_num_heuristics = NUM_HEURISTIC_FEATURES_FULL if full_heuristics else NUM_HEURISTIC_FEATURES

    # Track seen game_ids for deduplication across databases
    seen_game_ids: set = set()
    games_processed = 0
    games_skipped = 0
    games_deduplicated = 0
    games_skipped_recovery = 0
    games_partial = 0  # Games where replay failed but samples were extracted using DB winner
    soft_target_hits = 0  # Moves where MCTS soft targets were successfully matched
    soft_target_misses = 0  # Moves with move_probs where no keys could be matched
    newest_game_time: str | None = None  # Track newest game timestamp for freshness metadata

    # Build query filters
    query_filters: dict[str, Any] = {
        "board_type": board_type,
        "num_players": num_players,
        "require_moves": require_moves,
    }
    if min_moves is not None:
        query_filters["min_moves"] = min_moves
    if max_moves is not None:
        query_filters["max_moves"] = max_moves

    # Optional parity cutoffs
    parity_cutoffs: dict[str, int] = {}
    if parity_fixtures_dir:
        fixtures_path = os.path.abspath(parity_fixtures_dir)
        if os.path.isdir(fixtures_path):
            for name in os.listdir(fixtures_path):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(fixtures_path, name)
                try:
                    with open(path, encoding="utf-8") as f:
                        fixture = json.load(f)
                except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
                    continue
                game_id = fixture.get("game_id")
                diverged_at = fixture.get("diverged_at")
                if not isinstance(game_id, str) or not isinstance(diverged_at, int) or diverged_at <= 0:
                    continue
                safe_max_move = diverged_at - 1
                prev = parity_cutoffs.get(game_id)
                if prev is None or safe_max_move < prev:
                    parity_cutoffs[game_id] = safe_max_move

    # Log filter configuration
    filter_desc = []
    if require_moves:
        filter_desc.append("require move data")
    if require_completed:
        filter_desc.append("completed games only")
    if min_moves is not None:
        filter_desc.append(f"min {min_moves} moves")
    if max_moves is not None:
        filter_desc.append(f"max {max_moves} moves")
    if exclude_recovery:
        filter_desc.append("excluding recovery games")
    if min_quality is not None:
        filter_desc.append(f"min quality score {min_quality:.2f}")
    if min_elo is not None:
        filter_desc.append(f"min generator Elo {min_elo:.0f}")
    if quality_tier is not None and quality_tier != "bootstrap":
        filter_desc.append(f"quality tier: {quality_tier}")
    if filter_desc:
        print(f"Quality filters: {', '.join(filter_desc)}")
    print(f"Value targets: {'rank-aware' if use_rank_aware_values else 'binary winner/loser'}")
    print(f"Processing {len(db_paths)} database(s) with deduplication")

    # Process each database
    for db_idx, db_path in enumerate(db_paths):
        if not os.path.exists(db_path):
            print(f"  [{db_idx+1}/{len(db_paths)}] Skipping missing: {db_path}")
            continue

        print(f"  [{db_idx+1}/{len(db_paths)}] Processing: {os.path.basename(db_path)}...")

        # Dec 27, 2025: Use retry logic for database lock errors during cluster sync
        try:
            db = _open_db_with_retry(db_path)
        except Exception as e:
            if _is_db_locked_error(e):
                print(f"    SKIPPED (database locked after retries): {e}")
            else:
                print(f"    Error opening database: {e}")
            continue

        # January 2026: Move data validation gate (Phase 6 Data Integrity)
        # Validate database has no orphan games (games with total_moves > 0 but no move data)
        # This prevents corrupt training data from games that were only partially written
        from app.db.move_data_validator import (
            MoveDataValidator,
            MoveDataValidationError,
            MIN_MOVES_REQUIRED,
        )

        validation_result = MoveDataValidator.validate_database(db_path, min_moves=MIN_MOVES_REQUIRED)
        if validation_result.invalid_count > 0:
            orphan_msg = (
                f"    Found {validation_result.invalid_count} orphan games "
                f"(games with insufficient move data) in {os.path.basename(db_path)}"
            )
            if fail_on_orphans:
                # Get the list of orphan game IDs for the exception
                try:
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    orphan_games = MoveDataValidator.get_games_without_moves(conn, MIN_MOVES_REQUIRED)
                    conn.close()
                    orphan_ids = [gid for gid, _ in orphan_games]
                except Exception:
                    orphan_ids = []  # Can't get IDs, use empty list

                raise MoveDataValidationError(
                    orphan_ids,
                    f"Cannot export {os.path.basename(db_path)}: "
                    f"{validation_result.invalid_count} games missing move data. "
                    f"Use --no-strict to skip validation and continue (NOT RECOMMENDED)."
                )
            else:
                # Warn and continue - orphan games will be filtered by require_moves
                print(f"{orphan_msg} (continuing due to --no-strict)")
        elif not validation_result.has_any_move_data:
            # Database has no move data in any format, skip entirely
            print(f"    SKIPPED (metadata-only database, no move data)")
            continue
        elif validation_result.has_inline_moves and not validation_result.has_game_moves_table:
            # Database has inline moves only - will use fallback parsing
            print(f"    Using inline moves: {validation_result.inline_moves_count} games with games.moves data")

        db_games = 0
        db_samples = 0
        db_dedup = 0

        # Get total game count for progress tracking (January 2026)
        try:
            db_total_games = db.get_game_count(
                board_type=board_type,
                num_players=num_players,
            )
        except Exception:
            db_total_games = 0  # Fallback if count fails
        progress_interval = max(100, db_total_games // 20) if db_total_games > 0 else 100  # Report ~20 times or every 100 games
        import time as _time_module
        _progress_start = _time_module.time()
        _last_progress_time = _progress_start

        for meta, initial_state, moves, game_move_probs in db.iterate_games_with_probs(**query_filters):
            game_id = meta.get("game_id")

            # Deduplication: skip if we've seen this game_id before
            if game_id in seen_game_ids:
                db_dedup += 1
                games_deduplicated += 1
                continue
            seen_game_ids.add(game_id)

            # Source filtering (December 2025 - Phase 5 Unified NN/NNUE training)
            # Filter games by source type (selfplay, gauntlet, tournament, etc.)
            if include_sources is not None or exclude_sources is not None:
                source_raw = str(meta.get("source", "") or "").lower()
                # Categorize source
                if "gauntlet" in source_raw:
                    game_source = "gauntlet"
                elif "tournament" in source_raw:
                    game_source = "tournament"
                elif "human" in source_raw:
                    game_source = "human"
                else:
                    game_source = "selfplay"  # Default category

                # Apply inclusion filter
                if include_sources is not None and "all" not in include_sources:
                    if game_source not in include_sources:
                        games_skipped += 1
                        continue

                # Apply exclusion filter
                if exclude_sources is not None and game_source in exclude_sources:
                    games_skipped += 1
                    continue

            # Quality tier filtering (January 2026 - Training Quality Fix)
            # Filter games by engine mode to exclude low-quality heuristic-only data
            if quality_tier is not None and quality_tier != "bootstrap":
                source_raw_qt = str(meta.get("source", "") or "").lower()
                # Determine engine mode from source
                if "gumbel" in source_raw_qt:
                    qt_engine_mode = "gumbel_mcts"
                elif "mcts" in source_raw_qt:
                    qt_engine_mode = "mcts"
                elif "mixed" in source_raw_qt:
                    qt_engine_mode = "mixed"
                elif "policy" in source_raw_qt:
                    qt_engine_mode = "policy_only"
                elif "descent" in source_raw_qt:
                    qt_engine_mode = "descent"
                elif "gpu" in source_raw_qt or "heuristic" in source_raw_qt:
                    qt_engine_mode = "heuristic"
                else:
                    qt_engine_mode = "unknown"

                # Apply quality tier filter
                # standard: Exclude heuristic-only data (no tree search)
                # quality: Only tree search engines (mcts, gumbel-mcts, mixed)
                # ultimate: Only gumbel-mcts (highest quality)
                if quality_tier == "standard":
                    # Exclude pure heuristic games (no tree search)
                    if qt_engine_mode == "heuristic":
                        games_skipped += 1
                        continue
                elif quality_tier == "quality":
                    # Only include tree search engines
                    if qt_engine_mode not in ("gumbel_mcts", "mcts", "mixed"):
                        games_skipped += 1
                        continue
                elif quality_tier == "ultimate":
                    # Only include gumbel-mcts (highest quality)
                    if qt_engine_mode != "gumbel_mcts":
                        games_skipped += 1
                        continue

            if require_completed:
                status = str(meta.get("game_status", ""))
                term = str(meta.get("termination_reason", ""))
                if status != "completed":
                    games_skipped += 1
                    continue
                if term and not (term.startswith("status:completed") or term == "env_done_flag"):
                    games_skipped += 1
                    continue

            # Quality scoring (December 2025) - always compute for weighted training
            game_quality_score = 1.0  # Default to max quality if scorer unavailable
            if HAS_QUALITY_SCORER:
                quality = compute_game_quality_from_params(
                    game_id=game_id or "unknown",
                    game_status=str(meta.get("game_status", "")),
                    winner=meta.get("winner"),
                    termination_reason=str(meta.get("termination_reason", "")),
                    total_moves=len(moves) if moves else 0,
                    board_type=board_type.value if hasattr(board_type, "value") else str(board_type),
                    source=str(meta.get("source", "")),
                )
                game_quality_score = quality.quality_score
                # Quality filtering: skip games below threshold
                if min_quality is not None and game_quality_score < min_quality:
                    games_skipped += 1
                    continue

            # Extract generator model Elo (January 2026 - quality-weighted sampling)
            # This is used both for Elo gating (filtering) and sample weighting.
            # Games from stronger models provide higher quality training data.
            # Try metadata_json first (new games), then legacy model_elo field
            game_generator_elo: float = 1500.0  # Default to baseline Elo
            metadata_json_raw = meta.get("metadata_json")
            if metadata_json_raw:
                try:
                    metadata_dict = json.loads(metadata_json_raw) if isinstance(metadata_json_raw, str) else metadata_json_raw
                    if metadata_dict.get("model_elo") is not None:
                        game_generator_elo = float(metadata_dict.get("model_elo"))
                except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
                    pass
            if game_generator_elo == 1500.0 and meta.get("model_elo") is not None:
                try:
                    game_generator_elo = float(meta.get("model_elo"))
                except (TypeError, ValueError):
                    pass

            # Elo gating (January 2026 - iterative strength improvement)
            # Filter games by the generating model's Elo rating. Games from weaker
            # models produce lower quality training data. Only train on data from
            # sufficiently strong models to ensure iterative improvement.
            if min_elo is not None:
                # Skip games from weak models (below min_elo threshold)
                if game_generator_elo < min_elo:
                    games_skipped += 1
                    continue

            # Extract victory type for balanced sampling (normalize to standard categories)
            victory_type_raw = str(meta.get("victory_type", meta.get("termination_reason", "unknown")))
            if "territory" in victory_type_raw.lower():
                victory_type = "territory"
            elif "elimination" in victory_type_raw.lower() or "ring" in victory_type_raw.lower():
                victory_type = "elimination"
            elif "lps" in victory_type_raw.lower() or "last_player" in victory_type_raw.lower():
                victory_type = "lps"
            elif "stalemate" in victory_type_raw.lower():
                victory_type = "stalemate"
            elif "timeout" in victory_type_raw.lower():
                victory_type = "timeout"
            else:
                victory_type = "other"

            # Extract engine mode from source for sample weighting
            # Gumbel MCTS games get higher weight in training
            source_raw = str(meta.get("source", "") or "")
            if "gumbel" in source_raw.lower():
                engine_mode = "gumbel_mcts"
            elif "mcts" in source_raw.lower():
                engine_mode = "mcts"
            elif "policy" in source_raw.lower():
                engine_mode = "policy_only"
            elif "descent" in source_raw.lower():
                engine_mode = "descent"
            elif "heuristic" in source_raw.lower():
                engine_mode = "heuristic"
            elif "gpu" in source_raw.lower():
                engine_mode = "heuristic"  # GPU selfplay uses heuristic
            else:
                engine_mode = "unknown"

            # Extract opponent ELO for ELO-weighted training (December 2025)
            # Use 1500.0 as default (baseline ELO) if not available
            opponent_elo = float(meta.get("opponent_elo", meta.get("model_elo", 1500.0)))

            # Extract opponent type for diversity tracking (December 2025)
            # Check both new column and legacy metadata locations
            opponent_type = meta.get("opponent_type") or ""
            if not opponent_type:
                # Infer from source/engine_mode for backward compatibility
                if "random" in source_raw.lower():
                    opponent_type = "random"
                elif "heuristic" in source_raw.lower():
                    opponent_type = "heuristic"
                elif "mcts" in source_raw.lower():
                    opponent_type = "mcts"
                elif "gumbel" in source_raw.lower():
                    opponent_type = "gumbel_mcts"
                elif "neural" in source_raw.lower() or "nn" in source_raw.lower():
                    opponent_type = "neural"
                else:
                    opponent_type = "unknown"

            # Extract game timestamp for freshness weighting (December 2025)
            # Convert to Unix epoch for compute_freshness_weight()
            game_timestamp: float = 0.0  # Default to epoch (will get max_weight)
            game_time_raw = meta.get("completed_at") or meta.get("created_at")
            if game_time_raw:
                try:
                    from datetime import datetime
                    if isinstance(game_time_raw, (int, float)):
                        game_timestamp = float(game_time_raw)
                    elif isinstance(game_time_raw, str):
                        # Try ISO format (most common)
                        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                            try:
                                dt = datetime.strptime(game_time_raw[:19], fmt[:19])
                                game_timestamp = dt.timestamp()
                                break
                            except ValueError:
                                continue
                except Exception:
                    pass  # Keep default of 0.0

            # Store database winner for partial game handling (December 2025)
            # This allows us to extract training samples even when replay fails partway
            db_winner = meta.get("winner")

            total_moves = meta.get("total_moves")
            if total_moves is None:
                total_moves = len(moves) if moves else 0
            total_moves = int(total_moves)
            if total_moves <= 0 or not moves:
                continue

            if exclude_recovery:
                has_recovery = any(
                    "recovery" in str(getattr(m, "type", "")).lower()
                    for m in moves
                )
                if has_recovery:
                    games_skipped_recovery += 1
                    continue

            max_safe_move_index: int | None = None
            if parity_cutoffs:
                cutoff = parity_cutoffs.get(game_id)
                if cutoff is not None:
                    max_safe_move_index = cutoff
                    if max_safe_move_index <= 0:
                        games_skipped += 1
                        continue

            # NOTE: We defer final_state computation until after incremental replay
            # to avoid slow db.get_state_at_move() call. The incremental replay will
            # give us the final state naturally.
            final_state_index = total_moves - 1
            if max_safe_move_index is not None:
                final_state_index = min(final_state_index, max_safe_move_index)

            num_players_in_game = len(initial_state.players)

            # Collect samples first, then compute values after we have final state
            game_samples: list[tuple[np.ndarray, np.ndarray, int, int, str]] = []
            history_frames: list[np.ndarray] = []
            samples_before = len(features_list)

            # Load pre-computed heuristics from database cache (Jan 2026)
            # This provides 10-20x speedup for --full-heuristics exports
            cached_heuristics: dict[int, np.ndarray] = {}
            if include_heuristics:
                try:
                    cached_heuristics = db.get_game_heuristics_batch(game_id)
                except Exception:
                    pass  # Fallback to computing on-the-fly

            # Use incremental state updates instead of replaying from scratch for each move.
            # This reduces complexity from O(n²) to O(n) per game.
            from app.game_engine import GameEngine

            current_state = initial_state
            replay_succeeded = True
            move_probs_key_map: dict[int, dict[str, int]] = {}  # move_index -> {key: policy_idx}
            for move_index, move in enumerate(moves):
                if max_safe_move_index is not None and move_index > max_safe_move_index:
                    break
                if max_move_index is not None and move_index > max_move_index:
                    break

                # state_before is the state BEFORE this move is applied
                state_before = current_state

                # Apply move to get next state (for next iteration)
                try:
                    # Use trace_mode=True for canonical replay behavior
                    current_state = GameEngine.apply_move(current_state, move, trace_mode=True)
                except Exception as e:
                    logger.debug(f"Skipping game {game_id} at move {move_index}: {e}")
                    replay_succeeded = False
                    break

                # Skip if not sampling this move
                if sample_every > 1 and (move_index % sample_every) != 0:
                    continue

                stacked, globals_vec = encode_state_with_history(
                    encoder, state_before, history_frames, history_length=history_length
                )

                # Use the same encoder path as encode_state_with_history for consistent shapes
                hex_encoder = getattr(encoder, "_hex_encoder", None)
                if hex_encoder is not None:
                    base_features, _ = hex_encoder.encode_state(state_before)
                else:
                    base_features, _ = encoder._extract_features(state_before)
                history_frames.append(base_features)
                if len(history_frames) > history_length + 1:
                    history_frames.pop(0)

                # Normalize hex board size and move coords for legacy data
                normalized_board = _normalize_hex_board_size(state_before.board)
                normalized_move = _normalize_hex_move_coords(move, board_type, 25 if board_type == BoardType.HEXAGONAL else initial_state.board.size)

                if use_board_aware_encoding:
                    idx = encode_move_for_board(normalized_move, normalized_board)
                else:
                    idx = encoder.encode_move(normalized_move, normalized_board)
                if idx == INVALID_MOVE_INDEX:
                    continue

                # Store sample with perspective for later value computation
                phase_str = (
                    str(state_before.current_phase.value)
                    if hasattr(state_before.current_phase, "value")
                    else str(state_before.current_phase)
                )
                # Extract move type for chain-aware weighting
                move_type_raw = getattr(move, "type", None)
                if hasattr(move_type_raw, "value"):
                    move_type_str = str(move_type_raw.value)
                else:
                    move_type_str = str(move_type_raw) if move_type_raw else "unknown"

                # Extract heuristic features for v5 training (if enabled)
                # Fast mode: 21 component scores (O(1) extraction)
                # Full mode: 49 weight decomposition features (O(50) extraction)
                # Jan 2026: Check DB cache first for 10-20x speedup
                heuristic_vec = None
                if include_heuristics:
                    expected_size = 49 if full_heuristics else 21
                    # Try cached heuristics first (loaded at start of game loop)
                    cached = cached_heuristics.get(move_index)
                    if cached is not None and len(cached) == expected_size:
                        heuristic_vec = cached
                    else:
                        # Compute on-the-fly (legacy games without cache)
                        try:
                            if full_heuristics:
                                heuristic_vec = extract_full_heuristic_features(
                                    state_before,
                                    player_number=state_before.current_player,
                                    normalize=True,
                                )  # Returns np.ndarray of shape (49,)
                            else:
                                heuristic_vec = extract_heuristic_features(
                                    state_before,
                                    player_number=state_before.current_player,
                                    eval_mode="full",
                                    normalize=True,
                                )  # Returns np.ndarray of shape (21,)
                        except Exception as e:
                            # Fallback to zeros if extraction fails
                            heuristic_vec = np.zeros(effective_num_heuristics, dtype=np.float32)
                            logger.debug(f"Heuristic extraction failed at move {move_index}: {e}")

                # Build key→policy_idx mapping for MCTS soft targets
                # Maps build_move_key format -> policy index for all legal moves.
                # DB keys (short or full repr) are normalized to build_move_key format at lookup time.
                if game_move_probs.get(move_index):
                    try:
                        valid_moves = GameEngine.get_valid_moves(state_before, state_before.current_player)
                        key_map: dict[str, int] = {}
                        norm_board = _normalize_hex_board_size(state_before.board)
                        board_size_for_norm = 25 if board_type == BoardType.HEXAGONAL else initial_state.board.size
                        for m in valid_moves:
                            short_key = build_move_key(m)
                            norm_m = _normalize_hex_move_coords(m, board_type, board_size_for_norm)
                            if use_board_aware_encoding:
                                p_idx = encode_move_for_board(norm_m, norm_board)
                            else:
                                p_idx = encoder.encode_move(norm_m, norm_board)
                            if p_idx != INVALID_MOVE_INDEX:
                                key_map[short_key] = p_idx
                        move_probs_key_map[move_index] = key_map
                    except Exception as e:
                        logger.debug(f"Failed to build move key map at move {move_index}: {e}")

                game_samples.append((
                    stacked, globals_vec, idx, state_before.current_player,
                    move_index, phase_str, move_type_str, heuristic_vec
                ))

            # Handle partial games (December 2025 fix)
            # Key insight: If replay fails but we collected samples AND have winner from DB,
            # we can still use those samples for training. This fixes the hex8_4p export
            # issue where chain capture FSM mismatches caused all games to be skipped.
            if not game_samples:
                games_skipped += 1
                continue

            # Now we have final_state = current_state from incremental replay
            final_state = current_state

            # Determine winner: prefer replayed final_state, fallback to database winner
            effective_winner = getattr(final_state, 'winner', None)
            if effective_winner is None or effective_winner == 0:
                effective_winner = db_winner

            # Skip games without a valid winner - these produce value=0 which corrupts training
            if effective_winner is None or effective_winner == 0:
                games_skipped += 1
                continue

            # For partial games, patch the winner into final_state for value computation
            if not replay_succeeded and effective_winner is not None:
                # Create a mock final state with the correct winner for value computation
                # This allows value_from_final_winner/ranking to work correctly
                final_state = type(final_state)(
                    **{**final_state.__dict__, 'winner': effective_winner}
                )
                games_partial += 1
                logger.debug(
                    f"Game {game_id}: Partial replay ({len(game_samples)} samples) "
                    f"using DB winner {effective_winner}"
                )

            # Compute values using the final replayed state
            if use_rank_aware_values:
                values_vec = np.asarray(
                    compute_multi_player_values(final_state, num_players=num_players_in_game),
                    dtype=np.float32,
                )
            else:
                values_vec = np.zeros(4, dtype=np.float32)
                # Iterate over all expected players (1 to num_players), not just
                # those remaining in final_state.players (which may exclude eliminated players)
                for player_num in range(1, num_players_in_game + 1):
                    base = value_from_final_winner(final_state, player_num)
                    values_vec[player_num - 1] = float(base)

            # Add all samples from this game with computed values
            # NOTE: For scalar value targets, we use the CURRENT PLAYER's perspective.
            # This matches the feature encoding (which uses current player's view)
            # and the inference code (which expects current player's value and
            # flips it if needed, see gumbel_mcts_ai.py lines 790-791).
            # For multi-player training, values_mp provides per-player values.
            for stacked, globals_vec, idx, perspective, move_index, phase_str, move_type_str, heuristic_vec in game_samples:
                # Use current player's perspective (stored in 'perspective' variable)
                if use_rank_aware_values:
                    value = value_from_final_ranking(
                        final_state, perspective=perspective, num_players=num_players
                    )
                else:
                    value = value_from_final_winner(final_state, perspective=perspective)

                features_list.append(stacked)
                globals_list.append(globals_vec)
                values_list.append(float(value))

                # Use soft targets from move_probs if available, otherwise 1-hot
                soft_probs = game_move_probs.get(move_index)
                if soft_probs:
                    # Look up pre-computed key→policy_idx mapping built during replay
                    # Normalize DB keys (short or full repr) to build_move_key format
                    key_map = move_probs_key_map.get(move_index, {})
                    soft_indices = []
                    soft_values = []
                    for move_key, prob in soft_probs.items():
                        normalized_key = _normalize_db_move_key(move_key)
                        policy_idx = key_map.get(normalized_key)
                        if policy_idx is not None and prob > 0:
                            soft_indices.append(policy_idx)
                            soft_values.append(float(prob))
                    if soft_indices:
                        # Normalize probabilities to sum to 1
                        total = sum(soft_values)
                        if total > 0:
                            soft_values = [v / total for v in soft_values]
                        policy_indices_list.append(np.array(soft_indices, dtype=np.int32))
                        policy_values_list.append(np.array(soft_values, dtype=np.float32))
                        soft_target_hits += 1
                    else:
                        # Fallback to 1-hot if no keys matched
                        policy_indices_list.append(np.array([idx], dtype=np.int32))
                        policy_values_list.append(np.array([1.0], dtype=np.float32))
                        soft_target_misses += 1
                else:
                    # No soft targets available - use 1-hot encoding
                    policy_indices_list.append(np.array([idx], dtype=np.int32))
                    policy_values_list.append(np.array([1.0], dtype=np.float32))

                values_mp_list.append(values_vec)
                num_players_list.append(num_players_in_game)
                move_numbers_list.append(move_index)
                total_game_moves_list.append(total_moves)
                phases_list.append(phase_str)
                victory_types_list.append(victory_type)
                engine_modes_list.append(engine_mode)
                move_types_list.append(move_type_str)
                opponent_elo_list.append(opponent_elo)
                generator_elo_list.append(game_generator_elo)
                opponent_types_list.append(opponent_type)
                quality_score_list.append(game_quality_score)
                timestamps_list.append(game_timestamp)

                # Add heuristic features if extracted
                if include_heuristics:
                    if heuristic_vec is not None:
                        heuristics_list.append(heuristic_vec)
                    else:
                        heuristics_list.append(np.zeros(NUM_HEURISTIC_FEATURES, dtype=np.float32))

            samples_added = len(features_list) - samples_before
            if samples_added > 0:
                db_games += 1
                db_samples += samples_added

                # Progress logging (January 2026)
                if db_total_games > 0 and db_games % progress_interval == 0:
                    _current_time = _time_module.time()
                    _elapsed = _current_time - _progress_start
                    _pct = (db_games / db_total_games) * 100
                    _rate = db_games / _elapsed if _elapsed > 0 else 0
                    _remaining = (db_total_games - db_games) / _rate if _rate > 0 else 0
                    print(f"    Progress: {db_games:,}/{db_total_games:,} games ({_pct:.1f}%), "
                          f"{db_samples:,} samples, {_rate:.1f} games/s, ~{_remaining:.0f}s remaining",
                          flush=True)
                    _last_progress_time = _current_time

                # Track newest game timestamp for freshness metadata
                # Prefer completed_at over created_at since we want when game ended
                game_time = meta.get("completed_at") or meta.get("created_at")
                if game_time:
                    game_time_str = str(game_time)
                    if newest_game_time is None or game_time_str > newest_game_time:
                        newest_game_time = game_time_str

            games_processed += 1
            if max_games is not None and games_processed >= max_games:
                break

        _db_elapsed = _time_module.time() - _progress_start
        print(f"    -> {db_games} games, {db_samples} samples, {db_dedup} deduplicated ({_db_elapsed:.1f}s)")

        if max_games is not None and games_processed >= max_games:
            print(f"  Reached max_games limit ({max_games})")
            break

    if not features_list:
        print(f"No samples generated from {len(db_paths)} database(s) "
              f"(board={board_type}, players={num_players}).")
        return

    # Stack into arrays
    features_arr = np.stack(features_list, axis=0).astype(np.float32)
    globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
    values_arr = np.array(values_list, dtype=np.float32)
    policy_indices_arr = np.array(policy_indices_list, dtype=object)
    policy_values_arr = np.array(policy_values_list, dtype=object)
    values_mp_arr = np.stack(values_mp_list, axis=0).astype(np.float32)
    num_players_arr = np.array(num_players_list, dtype=np.int32)
    move_numbers_arr = np.array(move_numbers_list, dtype=np.int32)
    total_game_moves_arr = np.array(total_game_moves_list, dtype=np.int32)
    phases_arr = np.array(phases_list, dtype=object)
    victory_types_arr = np.array(victory_types_list, dtype=object)  # For balanced sampling
    engine_modes_arr = np.array(engine_modes_list, dtype=object)  # For source-based sample weighting
    move_types_arr = np.array(move_types_list, dtype=object)  # For chain-aware sample weighting
    opponent_elo_arr = np.array(opponent_elo_list, dtype=np.float32)  # For ELO-weighted training
    generator_elo_arr = np.array(generator_elo_list, dtype=np.float32)  # For quality-weighted sampling (Jan 2026)
    opponent_types_arr = np.array(opponent_types_list, dtype=object)  # For opponent diversity tracking
    quality_score_arr = np.array(quality_score_list, dtype=np.float32)  # For quality-weighted training

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    write_mp = True
    if os.path.exists(output_path) and not append:
        archived = f"{output_path}.archived_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            os.rename(output_path, archived)
            print(f"[export] archived existing output -> {archived}", file=sys.stderr)
        except OSError as exc:
            print(f"[export] Warning: failed to archive {output_path}: {exc}", file=sys.stderr)

    if os.path.exists(output_path) and append:
        try:
            with np.load(output_path, allow_pickle=True) as data:
                if "features" in data:
                    existing_features = data["features"]
                    existing_globals = data["globals"]
                    existing_values = data["values"]
                    existing_policy_indices = data["policy_indices"]
                    existing_policy_values = data["policy_values"]

                    has_mp = "values_mp" in data and "num_players" in data
                    if has_mp:
                        existing_values_mp = data["values_mp"]
                        existing_num_players = data["num_players"]
                        values_mp_arr = np.concatenate([existing_values_mp, values_mp_arr], axis=0)
                        num_players_arr = np.concatenate([existing_num_players, num_players_arr], axis=0)
                    else:
                        write_mp = False

                    features_arr = np.concatenate([existing_features, features_arr], axis=0)
                    globals_arr = np.concatenate([existing_globals, globals_arr], axis=0)
                    values_arr = np.concatenate([existing_values, values_arr], axis=0)
                    policy_indices_arr = np.concatenate([existing_policy_indices, policy_indices_arr], axis=0)
                    policy_values_arr = np.concatenate([existing_policy_values, policy_values_arr], axis=0)
                    # Handle victory_types if present in existing data
                    if "victory_types" in data:
                        existing_victory_types = data["victory_types"]
                        victory_types_arr = np.concatenate([existing_victory_types, victory_types_arr], axis=0)
                    print(f"Appended to existing dataset at {output_path}; new total samples: {values_arr.shape[0]}")
        except Exception as exc:
            print(f"Warning: failed to append to existing {output_path}: {exc}")

    # Phase 5: Compute additional metadata for validation
    # Compute max policy index across all samples
    max_policy_index = 0
    for indices in policy_indices_list:
        if len(indices) > 0:
            max_policy_index = max(max_policy_index, max(indices))

    # Determine encoder type for hex boards
    effective_encoder = encoder_version if encoder_version in ("v2", "v3") else "v3"
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        encoder_type_str = f"hex_{effective_encoder}"
    else:
        encoder_type_str = f"square_v{feature_version}"

    # Get policy size from constants
    from app.ai.neural_net.constants import get_policy_size_for_board
    policy_size = get_policy_size_for_board(board_type)

    # VALIDATION: Ensure policy indices are within valid range for the encoding type
    encoding_label = "board_aware" if use_board_aware_encoding else "legacy_max_n"
    if use_board_aware_encoding and max_policy_index >= policy_size:
        raise ValueError(
            f"POLICY ENCODING VALIDATION FAILED!\n"
            f"  Claimed encoding: {encoding_label}\n"
            f"  Max policy index in data: {max_policy_index}\n"
            f"  Expected policy size for {board_type.value}: {policy_size}\n\n"
            f"This means the export produced indices outside the board-aware action space.\n"
            f"Check that encode_move_for_board() is being called correctly."
        )
    elif not use_board_aware_encoding and max_policy_index < policy_size:
        # Warning: using legacy encoding but indices happen to fit in board-aware space
        # This is unusual but not an error
        print(f"[INFO] Legacy encoding used but max_index ({max_policy_index}) < board policy_size ({policy_size})")
        print(f"       Consider using --board-aware-encoding for better model compatibility.")

    save_kwargs = {
        "features": features_arr,
        "globals": globals_arr,
        "values": values_arr,
        "policy_indices": policy_indices_arr,
        "policy_values": policy_values_arr,
        "board_type": np.asarray(board_type.value),
        "board_size": np.asarray(int(features_arr.shape[-1])),
        "history_length": np.asarray(int(history_length)),
        "feature_version": np.asarray(int(feature_version)),
        "policy_encoding": np.asarray("board_aware" if use_board_aware_encoding else "legacy_max_n"),
        "move_numbers": move_numbers_arr,
        "total_game_moves": total_game_moves_arr,
        "phases": phases_arr,
        "victory_types": victory_types_arr,  # For victory-type-balanced sampling
        "engine_modes": engine_modes_arr,  # For source-based sample weighting (Gumbel 3x weight)
        "move_types": move_types_arr,  # For chain-aware sample weighting
        "opponent_elo": opponent_elo_arr,  # For ELO-weighted training (December 2025)
        "generator_elo": generator_elo_arr,  # For quality-weighted sampling by generator strength (January 2026)
        "quality_score": quality_score_arr,  # For quality-weighted training (December 2025)
        "opponent_types": opponent_types_arr,  # For opponent diversity tracking (December 2025)
    }

    # Add heuristic features if extracted (December 2025, v5 training)
    if include_heuristics and heuristics_list:
        heuristics_arr = np.stack(heuristics_list, axis=0).astype(np.float32)
        save_kwargs["heuristics"] = heuristics_arr
        save_kwargs["num_heuristic_features"] = np.asarray(int(effective_num_heuristics))
        save_kwargs["heuristic_mode"] = np.asarray("full" if full_heuristics else "fast")
        mode_str = "full (49)" if full_heuristics else "fast (21)"
        print(f"Including heuristic features: {heuristics_arr.shape} ({mode_str} features per sample)")

    # Add quality-weighted sample weights (December 2025)
    # Combines source quality (Gumbel 3x, MCTS 3x) with data freshness (half-life 3 days)
    if quality_weighted:
        if HAS_SOURCE_WEIGHTING:
            timestamps_arr = np.array(timestamps_list, dtype=np.float64)
            sample_weights = compute_combined_weights(
                engine_modes=engine_modes_arr,
                timestamps=timestamps_arr,
                source_config=None,  # Use defaults (Gumbel 3x, medium 1.5x, base 1x)
                freshness_config=None,  # Use defaults (half_life=3 days)
                normalize=True,
            )
            save_kwargs["sample_weights"] = sample_weights.astype(np.float32)
            save_kwargs["timestamps"] = timestamps_arr
            # Log weight distribution
            print(f"Quality-weighted sampling enabled:")
            print(f"  Sample weights: mean={sample_weights.mean():.3f}, "
                  f"std={sample_weights.std():.3f}, "
                  f"min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")
        else:
            print("[WARNING] --quality-weighted requested but source_weighting module unavailable")

    # Phase 5 Metadata
    save_kwargs.update({
        # Phase 5 Metadata: Additional fields for training compatibility validation
        "encoder_type": np.asarray(encoder_type_str),
        "encoder_version": np.asarray(effective_encoder),  # v2 or v3 - for model selection
        "in_channels": np.asarray(int(features_arr.shape[1])),  # Actual channel count for validation
        "policy_size": np.asarray(int(policy_size)),
        "max_policy_index": np.asarray(int(max_policy_index)),
        "policies_normalized": np.asarray(True),  # All policy values sum to 1.0
        "export_version": np.asarray("2.1"),  # Mark as having V3 encoder metadata
    })
    if write_mp:
        save_kwargs.update({"values_mp": values_mp_arr, "num_players": num_players_arr})

    # Validate spatial dimensions match expected board type
    actual_spatial = features_arr.shape[-1]
    expected_spatial_sizes = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEX8: 9,
        BoardType.HEXAGONAL: 25,
    }
    expected_spatial = expected_spatial_sizes.get(board_type)
    if expected_spatial is not None and actual_spatial != expected_spatial:
        raise ValueError(
            f"========================================\n"
            f"SPATIAL DIMENSION MISMATCH IN EXPORT\n"
            f"========================================\n"
            f"Board type: {board_type.name}\n"
            f"Feature spatial size: {actual_spatial}×{actual_spatial}\n"
            f"Expected spatial size: {expected_spatial}×{expected_spatial}\n\n"
            f"This indicates encoder misconfiguration.\n"
            f"Check that the encoder is using correct board_size.\n"
            f"========================================"
        )

    # Add spatial_size to metadata for training validation
    save_kwargs["spatial_size"] = np.asarray(int(actual_spatial))

    # Add freshness metadata for training data sync (December 2025)
    # This allows TrainingDataSyncDaemon to pick fresh data over stale data
    from datetime import datetime, timezone
    export_time = datetime.now(tz=timezone.utc).isoformat()
    save_kwargs["metadata"] = np.asarray({
        "export_time": export_time,
        "newest_game_time": newest_game_time,  # ISO format string or None
        "game_count": games_processed,
        "sample_count": len(features_arr),
    })

    # Add data checksums for integrity verification (December 2025)
    save_kwargs = embed_checksums_in_save_kwargs(save_kwargs)

    # Dec 28, 2025: Check disk space before writing to prevent partial/corrupted files
    has_space, space_msg = _check_disk_space_for_export(output_path, save_kwargs)
    if not has_space:
        raise IOError(
            f"[DISK SPACE ERROR] {space_msg}\n"
            f"Export aborted to prevent partial/corrupted NPZ file.\n"
            f"Free up disk space and retry."
        )
    print(f"[Disk Space] {space_msg}")

    # Feb 2026: Atomic write — write to temp file, validate, then rename.
    # Prevents corrupt NPZ files when process is killed mid-write.
    import tempfile as _tempfile
    _fd, _tmp_path = _tempfile.mkstemp(suffix=".npz", dir=str(Path(output_path).parent))
    os.close(_fd)
    try:
        np.savez_compressed(_tmp_path, **save_kwargs)

        # December 2025: Validate exported NPZ structure before declaring success
        # This catches corruption issues like the 22 billion element array incident
        try:
            from app.coordination.npz_validation import validate_npz_structure

            validation_result = validate_npz_structure(Path(_tmp_path))
            if not validation_result.valid:
                print(f"[NPZ VALIDATION FAILED] Export produced corrupted file!")
                for error in validation_result.errors:
                    print(f"  ERROR: {error}")
                for warning in validation_result.warnings:
                    print(f"  WARNING: {warning}")
                os.unlink(_tmp_path)
                return None  # Signal failure
            else:
                print(f"[NPZ VALIDATION] Passed: {validation_result.sample_count} samples, "
                      f"{len(validation_result.array_shapes)} arrays")
        except ImportError:
            print("[NPZ VALIDATION] Skipped (npz_validation module not available)")
        except Exception as e:
            print(f"[NPZ VALIDATION] Warning: Validation check failed: {e}")

        # Atomic rename — either the old file stays or the new one replaces it
        os.replace(_tmp_path, str(output_path))
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        if os.path.exists(_tmp_path):
            os.unlink(_tmp_path)
        raise

    # Log engine mode distribution for sample weighting visibility
    from collections import Counter
    mode_counts = Counter(engine_modes_list)
    total_samples = len(engine_modes_list)
    gumbel_count = mode_counts.get("gumbel_mcts", 0) + mode_counts.get("mcts", 0)
    gumbel_pct = 100 * gumbel_count / total_samples if total_samples > 0 else 0

    print(f"Exported {features_arr.shape[0]} samples from {games_processed} games "
          f"({games_deduplicated} deduplicated, {games_partial} partial) into {output_path}")
    if games_partial > 0:
        print(f"  Partial games: {games_partial} games had replay errors but samples extracted using DB winner")
    print(f"Engine modes: {dict(mode_counts)} | Gumbel/MCTS: {gumbel_pct:.1f}% (3x weight in training)")
    if soft_target_hits > 0 or soft_target_misses > 0:
        total_soft = soft_target_hits + soft_target_misses
        pct = 100.0 * soft_target_hits / total_soft if total_soft > 0 else 0
        print(f"  MCTS soft targets: {soft_target_hits}/{total_soft} matched ({pct:.1f}%), {soft_target_misses} fell back to 1-hot")

    # Phase 4A.4: Register NPZ file with ClusterManifest for cluster-wide discovery
    try:
        import socket
        from app.distributed.cluster_manifest import get_cluster_manifest

        manifest = get_cluster_manifest()
        file_size = os.path.getsize(output_path)
        node_id = socket.gethostname()

        manifest.register_npz(
            npz_path=str(output_path),
            node_id=node_id,
            board_type=board_type.value,
            num_players=num_players,
            sample_count=int(features_arr.shape[0]),
            file_size=file_size,
        )
        print(f"[ClusterManifest] Registered NPZ: {output_path} ({file_size // 1024 // 1024}MB, {features_arr.shape[0]} samples)")
    except Exception as e:
        # Don't fail export if manifest registration fails
        print(f"[ClusterManifest] Warning: Failed to register NPZ: {e}")


def export_replay_dataset(
    db_path: str,
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    feature_version: int = 2,
    sample_every: int = 1,
    max_games: int | None = None,
    require_completed: bool = False,
    min_moves: int | None = None,
    max_moves: int | None = None,
    max_move_index: int | None = None,
    use_rank_aware_values: bool = True,
    parity_fixtures_dir: str | None = None,
    exclude_recovery: bool = False,
    use_board_aware_encoding: bool = True,  # Default to board-aware encoding (not legacy MAX_N)
    append: bool = False,
    encoder_version: str = "default",
    require_moves: bool = True,
    min_quality: float | None = None,
    min_elo: float | None = None,  # January 2026: Elo-gated training
    fail_on_orphans: bool = True,
    quality_tier: str | None = None,  # January 2026: Quality tier filtering
) -> None:
    """
    Export training samples from a single GameReplayDB into an NPZ dataset.

    This is a convenience wrapper around export_replay_dataset_multi for
    single-database exports. For multi-source exports with deduplication,
    use export_replay_dataset_multi directly.
    """
    # Delegate to multi-source function with single DB
    export_replay_dataset_multi(
        db_paths=[db_path],
        board_type=board_type,
        num_players=num_players,
        output_path=output_path,
        history_length=history_length,
        feature_version=feature_version,
        sample_every=sample_every,
        max_games=max_games,
        require_completed=require_completed,
        min_moves=min_moves,
        max_moves=max_moves,
        max_move_index=max_move_index,
        use_rank_aware_values=use_rank_aware_values,
        parity_fixtures_dir=parity_fixtures_dir,
        exclude_recovery=exclude_recovery,
        use_board_aware_encoding=use_board_aware_encoding,
        append=append,
        require_moves=require_moves,
        encoder_version=encoder_version,
        min_quality=min_quality,
        min_elo=min_elo,
        fail_on_orphans=fail_on_orphans,
        quality_tier=quality_tier,
    )


def _fetch_remote_databases(
    board_type: str,
    num_players: int,
    include_remote: bool,
    include_s3: bool,
    include_owc: bool,
    target_dir: str,
) -> list[str]:
    """Fetch game databases from remote sources.

    January 2026: Enables cluster-wide data aggregation for export.

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)
        include_remote: Fetch from P2P cluster nodes
        include_s3: Fetch from S3
        include_owc: Fetch from OWC drive
        target_dir: Directory to store fetched databases

    Returns:
        List of paths to fetched database files
    """
    if not (include_remote or include_s3 or include_owc):
        return []

    try:
        import asyncio
        from pathlib import Path
        from app.utils.remote_game_fetcher import RemoteGameFetcher, FetchConfig

        config_key = f"{board_type}_{num_players}p"
        print(f"[REMOTE] Fetching databases from remote sources for {config_key}...")

        # Configure fetcher
        config = FetchConfig(
            fetch_from_p2p=include_remote,
            fetch_from_s3=include_s3,
            fetch_from_owc=include_owc,
            default_target_dir=Path(target_dir),
        )
        fetcher = RemoteGameFetcher(config)

        # Run async fetch
        async def do_fetch():
            return await fetcher.fetch_all_for_config(
                config_key=config_key,
                target_dir=Path(target_dir),
            )

        paths = asyncio.run(do_fetch())

        fetched_paths = [str(p) for p in paths]
        if fetched_paths:
            print(f"[REMOTE] Fetched {len(fetched_paths)} databases from remote sources")
            for p in fetched_paths:
                print(f"  {p}")
        else:
            print("[REMOTE] No remote databases found")

        return fetched_paths

    except ImportError as e:
        print(f"[REMOTE] RemoteGameFetcher not available: {e}")
        return []
    except Exception as e:
        print(f"[REMOTE] Error fetching remote databases: {e}")
        return []


def _check_remote_data_sources(board_type: str, num_players: int, db_paths: list[str]) -> None:
    """Check and log remote data source availability (informational only).

    This is a legacy function that logs available remote sources.
    Use --include-remote/--include-s3/--include-owc to actually fetch data.
    """
    try:
        import asyncio
        from app.utils.unified_game_aggregator import get_unified_game_aggregator

        async def check():
            aggregator = get_unified_game_aggregator()
            counts = await aggregator.get_total_games(
                board_type, num_players,
                include_remote=True,
                include_s3=True,
                include_owc=True,
            )
            return counts

        counts = asyncio.run(check())

        # Only log if there's significant remote data not in local paths
        local_count = counts.sources.get("local", 0)
        remote_total = counts.total_games - local_count

        if remote_total > 0:
            print(f"\n[INFO] Additional games available from remote sources:")
            for source, count in counts.sources.items():
                if source != "local" and count > 0:
                    print(f"  {source}: {count:,} games")
            print(f"  Use --include-remote/--include-s3/--include-owc to fetch\n")

    except ImportError:
        pass  # Aggregator not available
    except Exception as e:
        print(f"[DEBUG] Error checking remote sources: {e}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NN training samples from existing GameReplayDB replays.",
    )
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        dest="db_paths",
        help=(
            "Path to a GameReplayDB SQLite file. Can be specified multiple times "
            "for multi-source export with automatic deduplication by game_id. "
            "Example: --db db1.db --db db2.db. "
            "Required unless --use-discovery is specified."
        ),
    )
    parser.add_argument(
        "--use-discovery",
        action="store_true",
        help=(
            "Use unified GameDiscovery to automatically find all databases "
            "for the specified board-type and num-players. "
            "When enabled, --db is optional."
        ),
    )
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help=(
            "Skip the remote data source check when using --use-discovery. "
            "This avoids file handle exhaustion with many databases (500+). "
            "Use this flag for local-only exports or when remote sources are unavailable."
        ),
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hex8", "hexagonal"],
        required=True,
        help="Board type to filter games by.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        required=True,
        help="Number of players to filter games by.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npz dataset.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append to an existing output NPZ if present (legacy behavior). "
            "Default is to archive any existing output and rebuild from scratch."
        ),
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of past feature frames to stack (default: 3).",
    )
    parser.add_argument(
        "--feature-version",
        type=int,
        default=2,
        help=(
            "Feature encoding version for global feature layout (default: 2). "
            "Use 1 to keep legacy hex globals without chain/FE flags."
        ),
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Use every Nth move as a training sample (default: 1 = every move).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to process (default: all).",
    )
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="Only include games that completed normally (not timeout/disconnect).",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=None,
        help="Minimum move count to include a game (filters out trivially short games).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum move count to include a game (filters out abnormally long games).",
    )
    parser.add_argument(
        "--min-elo",
        type=float,
        default=None,
        help=(
            "Minimum generator model Elo to include a game. Games from weaker models are "
            "filtered out. This implements Elo-gated training for quality-focused improvement. "
            "Default: None (include all games). Recommended: 1050 for iterative improvement."
        ),
    )
    parser.add_argument(
        "--max-move-index",
        type=int,
        default=None,
        help=(
            "Maximum move index to sample within each game (limits replay depth). "
            "Use this to speed up export for games with many moves by only sampling "
            "early-game positions where replay is fast. E.g., --max-move-index 100 "
            "only samples moves 0-100 regardless of game length."
        ),
    )
    parser.add_argument(
        "--no-rank-aware-values",
        action="store_true",
        help="Use binary winner/loser values instead of rank-aware values for multiplayer.",
    )
    parser.add_argument(
        "--parity-fixtures-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing TS↔Python replay parity fixtures "
            "(as produced by scripts/check_ts_python_replay_parity.py "
            "with --emit-fixtures-dir). When provided, export only uses "
            "pre-divergence states per game."
        ),
    )
    parser.add_argument(
        "--exclude-recovery",
        action="store_true",
        help=(
            "Exclude games that contain recovery slide moves. "
            "Use this for training data purity when recovery rules have changed."
        ),
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help=(
            "Minimum quality score (0.0-1.0) to include a game. Uses UnifiedQualityScorer "
            "to evaluate game quality based on completion status, move count, and source. "
            "Recommended: 0.5 for adequate+ quality, 0.7 for good+ quality."
        ),
    )
    parser.add_argument(
        "--quality-tier",
        type=str,
        choices=["bootstrap", "standard", "quality", "ultimate"],
        default=None,
        help=(
            "Filter games by quality tier based on engine mode (Jan 2026). "
            "bootstrap: All data (no filtering). "
            "standard: Exclude gpu_heuristic and heuristic-only (no tree search). "
            "quality: Only mcts/gumbel-mcts/mixed games (tree search enabled). "
            "ultimate: Only gumbel-mcts games (highest quality data)."
        ),
    )
    parser.add_argument(
        "--board-aware-encoding",
        action="store_true",
        default=True,  # Now the default - flag kept for backwards compatibility
        help=(
            "[DEFAULT] Use board-specific policy encoding (compact action space). "
            "square8: 7000 actions, square19: 67000 actions. "
            "This is now the default. Use --legacy-max-n-encoding for old behavior."
        ),
    )
    parser.add_argument(
        "--legacy-max-n-encoding",
        action="store_true",
        help=(
            "DEPRECATED: Use legacy MAX_N=19 policy encoding (~59000 actions for all boards). "
            "This produces training data incompatible with v3/v4 models. "
            "Only use for reproducing old experiments."
        ),
    )
    parser.add_argument(
        "--encoder-version",
        type=str,
        choices=["default", "v2", "v3"],
        default="default",
        help=(
            "Encoder version for hex boards. "
            "'default' maps to 'v3' (recommended), "
            "'v2' uses HexStateEncoder (10 channels for HexNeuralNet_v2), "
            "'v3' uses HexStateEncoderV3 (16 channels for HexNeuralNet_v3). "
            "Hex boards ALWAYS use specialized encoders for consistent shapes. "
            "NOTE: Use --canonical-model to auto-detect from existing model."
        ),
    )
    parser.add_argument(
        "--canonical-model",
        type=str,
        default=None,
        help=(
            "Path to canonical model to match encoder version. "
            "Auto-detects the correct encoder version from the model's architecture. "
            "This ensures training data is compatible with the target model. "
            "Example: --canonical-model models/canonical_hex8_2p.pth. "
            "If not specified, falls back to config-based canonical model lookup."
        ),
    )
    parser.add_argument(
        "--no-require-moves",
        action="store_true",
        help=(
            "Disable the require_moves filter. By default, only games with "
            "actual move data in the game_moves table are included. Use this "
            "flag to include games without move data (they will be skipped "
            "anyway, but without this optimization the query is slower)."
        ),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "Enable incremental export caching. Skips export if source DBs "
            "haven't changed since last export. Significantly speeds up "
            "repeated training runs."
        ),
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help=(
            "Force re-export even if cache indicates no changes. "
            "Use with --use-cache to rebuild cache."
        ),
    )
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help=(
            "Disable parallel encoding and use single-threaded mode. "
            "Default is parallel mode which is 10-20x faster on multi-core systems."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for parallel mode (default: CPU count - 1). "
            "Set to 1 for single-threaded mode (same as --single-threaded)."
        ),
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help=(
            "Allow exporting from non-canonical DBs even when the output name "
            "starts with canonical_. Use for legacy/experimental datasets only."
        ),
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help=(
            "Allow DBs marked pending_gate in TRAINING_DATA_REGISTRY.md "
            "(still requires gate summary to be canonical_ok when present)."
        ),
    )
    parser.add_argument(
        "--strict/--no-strict",
        dest="strict",
        action="store_true",
        default=True,
        help=(
            "Strict mode (default: on): Reject databases with games that have "
            "insufficient move data (< 5 moves). This prevents exporting training "
            "data from databases with incomplete or corrupted game records. "
            "Use --no-strict to export anyway with a warning (NOT RECOMMENDED)."
        ),
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict mode - allow databases with incomplete move data (NOT RECOMMENDED).",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to TRAINING_DATA_REGISTRY.md (default: repo root)",
    )
    parser.add_argument(
        "--include-heuristics",
        action="store_true",
        help=(
            "Include heuristic features for v5/v5-heavy model training. "
            "Extracts 21 component scores per state from the heuristic AI "
            "(territory, threats, connectivity, mobility, etc). Fast O(1) extraction. "
            "Adds 'heuristics' array to NPZ output. Works in both parallel and single-threaded mode."
        ),
    )
    parser.add_argument(
        "--full-heuristics",
        action="store_true",
        help=(
            "Use full 49-feature heuristic extraction (maximum strength mode). "
            "Requires --include-heuristics. Uses linear weight decomposition "
            "instead of fast component scores. Slower (O(50) vs O(1) per state) "
            "but captures all heuristic knowledge including line patterns, swap "
            "evaluation, and recovery mechanics. Best for training maximum-strength models."
        ),
    )
    parser.add_argument(
        "--quality-weighted",
        action="store_true",
        help=(
            "Export sample weights based on source quality and data freshness. "
            "Adds 'sample_weights' array to NPZ output for use in weighted training. "
            "High-quality engines (gumbel, mcts) get 3x weight, recent games get "
            "higher weight via exponential decay (half-life=3 days). "
            "December 2025: Improves training convergence by 15-25%%."
        ),
    )
    # Source filtering flags (December 2025 - Phase 5 Unified NN/NNUE training)
    parser.add_argument(
        "--include-gauntlet",
        action="store_true",
        help=(
            "Include games from gauntlet evaluations in addition to selfplay. "
            "Gauntlet games have higher search quality (longer time controls) "
            "and diverse opponents. Adds ~5-20%% more training data per config. "
            "December 2025: Enables training from evaluation games."
        ),
    )
    parser.add_argument(
        "--include-tournaments",
        action="store_true",
        help=(
            "Include games from tournament play in addition to selfplay. "
            "Tournament games feature strong opponents and varied playstyles. "
            "Use for training robust models that generalize well. "
            "December 2025: Part of unified NN/NNUE training pipeline."
        ),
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help=(
            "Include games with human players from sandbox/lobby play. "
            "Human games are weighted 3x in training (HIGH_QUALITY tier). "
            "Use this to leverage human gameplay for NN improvement. "
            "January 2026: Part of human gameplay training pipeline."
        ),
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of source types to include. "
            "Options: selfplay, gauntlet, tournament, human, all. "
            "Default (None) includes only selfplay. "
            "Example: --sources selfplay,gauntlet,tournament"
        ),
    )
    parser.add_argument(
        "--exclude-sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of source types to exclude. "
            "Applied after --sources filtering. "
            "Example: --exclude-sources human"
        ),
    )
    # January 2026: Remote source fetching flags
    parser.add_argument(
        "--include-remote",
        action="store_true",
        help=(
            "Fetch game databases from P2P cluster nodes before export. "
            "Uses RemoteGameFetcher to download from other nodes in the cluster. "
            "Fetched DBs are stored in data/games/fetched/ and included in export. "
            "January 2026: Enables cluster-wide data aggregation."
        ),
    )
    parser.add_argument(
        "--include-s3",
        action="store_true",
        help=(
            "Fetch game databases from AWS S3 before export. "
            "Downloads from s3://ringrift-models-20251214/consolidated/games/. "
            "Requires AWS credentials configured. "
            "January 2026: Enables S3 data integration."
        ),
    )
    parser.add_argument(
        "--include-owc",
        action="store_true",
        help=(
            "Fetch game databases from OWC external drive before export. "
            "Downloads from mac-studio:/Volumes/RingRift-Data/selfplay_repository/. "
            "Requires SSH key access to OWC host. "
            "January 2026: Enables OWC data integration."
        ),
    )
    parser.add_argument(
        "--fetch-target-dir",
        type=str,
        default="data/games/fetched",
        help="Directory to store fetched remote databases (default: data/games/fetched)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Check for autonomous mode (December 2025)
    # In autonomous mode, allow non-canonical sources and pending gate DBs
    try:
        from app.config.env import env
        if env.autonomous_mode:
            if not args.allow_noncanonical:
                args.allow_noncanonical = True
                print("[AUTONOMOUS] Enabled --allow-noncanonical via RINGRIFT_AUTONOMOUS_MODE")
            if not args.allow_pending_gate:
                args.allow_pending_gate = True
                print("[AUTONOMOUS] Enabled --allow-pending-gate via RINGRIFT_AUTONOMOUS_MODE")
    except ImportError:
        pass  # env module not available, skip autonomous mode check

    # Feb 2026: Also check RINGRIFT_ALLOW_PENDING_GATE env var (set by auto_export_daemon)
    if not args.allow_pending_gate and os.environ.get("RINGRIFT_ALLOW_PENDING_GATE", "").lower() in ("true", "1", "yes"):
        args.allow_pending_gate = True

    board_type = BOARD_TYPE_MAP[args.board_type]
    if args.history_length < 0:
        raise ValueError("--history-length must be >= 0")
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

    # Handle encoding flag logic: --legacy-max-n-encoding overrides default
    use_board_aware = not getattr(args, 'legacy_max_n_encoding', False)
    if not use_board_aware:
        print("[WARNING] Using deprecated legacy_max_n encoding - data will be incompatible with v3/v4 models")
    # Override args.board_aware_encoding for downstream code
    args.board_aware_encoding = use_board_aware

    # January 2026: Auto-detect encoder version from canonical model (FAIL-FAST)
    # This prevents architecture mismatches between training data and target model
    canonical_model_path = getattr(args, 'canonical_model', None)
    if canonical_model_path is None:
        # Try to find canonical model automatically
        config_key = f"{args.board_type}_{args.num_players}p"
        auto_canonical = f"models/canonical_{config_key}.pth"
        if os.path.exists(auto_canonical):
            canonical_model_path = auto_canonical
            print(f"[ARCHITECTURE] Auto-detected canonical model: {canonical_model_path}")

    if canonical_model_path:
        try:
            from app.ai.neural_net.architecture_registry import (
                get_encoder_version_from_checkpoint,
                validate_export_architecture_match,
            )
            detected_version = get_encoder_version_from_checkpoint(canonical_model_path)
            if detected_version:
                print(f"[ARCHITECTURE] Canonical model requires encoder: {detected_version}")
                # If user explicitly specified a conflicting version, fail fast
                if args.encoder_version not in ("default", detected_version):
                    print(f"[ARCHITECTURE ERROR] Encoder mismatch!")
                    print(f"  Canonical model: {canonical_model_path}")
                    print(f"  Model requires: {detected_version}")
                    print(f"  You specified: {args.encoder_version}")
                    print(f"  SOLUTION: Use --encoder-version {detected_version} or remove --encoder-version")
                    return 1
                # Auto-set the encoder version to match canonical model
                args.encoder_version = detected_version
                print(f"[ARCHITECTURE] Using encoder version: {args.encoder_version}")
        except Exception as e:
            print(f"[ARCHITECTURE WARNING] Could not detect encoder from {canonical_model_path}: {e}")
            print("  Proceeding with specified encoder version...")

    # Handle database discovery
    db_paths = args.db_paths or []

    if args.use_discovery:
        if not HAS_GAME_DISCOVERY:
            raise RuntimeError("--use-discovery requires GameDiscovery module")
        print(f"[DISCOVERY] Finding databases for {args.board_type} {args.num_players}p...")
        discovery = GameDiscovery()
        discovered_dbs = discovery.find_databases_for_config(args.board_type, args.num_players)
        for db_info in discovered_dbs:
            if db_info.game_count > 0 and str(db_info.path) not in db_paths:
                db_paths.append(str(db_info.path))
                print(f"  Found: {db_info.path} ({db_info.game_count:,} games)")
        if not db_paths:
            print("[DISCOVERY] No databases found for this configuration")
            return 1
        print(f"[DISCOVERY] Total: {len(db_paths)} databases")

        # Check remote sources for additional data (January 2026 enhancement)
        # Skip if --skip-remote-check is set (avoids file handle exhaustion with 500+ DBs)
        if not getattr(args, 'skip_remote_check', False):
            _check_remote_data_sources(args.board_type, args.num_players, db_paths)
        else:
            print("[DISCOVERY] Skipping remote data source check (--skip-remote-check)")

    # January 2026: Fetch from remote sources if requested
    if getattr(args, 'include_remote', False) or getattr(args, 'include_s3', False) or getattr(args, 'include_owc', False):
        remote_dbs = _fetch_remote_databases(
            board_type=args.board_type,
            num_players=args.num_players,
            include_remote=getattr(args, 'include_remote', False),
            include_s3=getattr(args, 'include_s3', False),
            include_owc=getattr(args, 'include_owc', False),
            target_dir=getattr(args, 'fetch_target_dir', 'data/games/fetched'),
        )
        for remote_db in remote_dbs:
            if remote_db not in db_paths:
                db_paths.append(remote_db)

    if not db_paths:
        raise ValueError("Either --db or --use-discovery must be specified")

    # Pre-export validation: Check each database for games with move data
    # This prevents silent export failures from databases without move data
    from app.training.data_quality import validate_database_for_export

    print("\n[PRE-EXPORT VALIDATION] Checking databases for move data...")
    valid_db_paths = []
    for db_path in db_paths:
        valid, msg = validate_database_for_export(
            db_path,
            board_type=args.board_type,
            num_players=args.num_players,
        )
        if not valid:
            print(f"  SKIP: {os.path.basename(db_path)}")
            print(f"        {msg}")
        else:
            if "WARNING" in msg:
                print(f"  WARN: {os.path.basename(db_path)}")
                print(f"        {msg}")
            else:
                print(f"  OK:   {os.path.basename(db_path)}")
            valid_db_paths.append(db_path)

    if not valid_db_paths:
        print("\n[ERROR] No valid databases found for export!")
        print("All databases either have no games or no move data in game_moves table.")
        print("\nCommon causes:")
        print("  1. JSONL-to-DB import didn't populate game_moves table")
        print("  2. Selfplay ran without move recording")
        print("\nTo fix, re-run selfplay with proper move recording, or use")
        print("scripts/jsonl_to_npz.py to convert JSONL directly to NPZ format.")
        return 1

    # Update db_paths to only include valid databases
    db_paths = valid_db_paths
    print(f"[PRE-EXPORT VALIDATION] {len(db_paths)} database(s) passed validation\n")

    # January 2026: Strict mode - validate move data completeness using MoveDataValidator
    # This enforces MIN_MOVES_REQUIRED across all export operations
    if args.strict:
        from app.db.move_data_validator import MIN_MOVES_REQUIRED, MoveDataValidator

        print(f"[STRICT MODE] Validating move data completeness (min {MIN_MOVES_REQUIRED} moves/game)...")
        strict_failures = []
        for db_path in db_paths:
            try:
                result = MoveDataValidator.validate_database(db_path)
                if not result.has_any_move_data:
                    strict_failures.append(
                        f"  {os.path.basename(db_path)}: No move data (metadata-only database)"
                    )
                elif result.has_inline_moves and not result.has_game_moves_table:
                    # Inline moves only - not a failure, just informational
                    print(f"  INFO: {os.path.basename(db_path)} ({result.inline_moves_count} games with inline moves)")
                elif result.invalid_count > 0:
                    strict_failures.append(
                        f"  {os.path.basename(db_path)}: {result.invalid_count}/{result.total_games} games "
                        f"have insufficient move data (<{MIN_MOVES_REQUIRED} moves)"
                    )
                else:
                    print(f"  OK:   {os.path.basename(db_path)} ({result.total_games} games, all have sufficient moves)")
            except Exception as e:
                strict_failures.append(f"  {os.path.basename(db_path)}: Validation error: {e}")

        if strict_failures:
            print("\n[STRICT MODE] FAILED - The following databases have incomplete move data:")
            for failure in strict_failures:
                print(failure)
            print(
                f"\nExport BLOCKED. Games without at least {MIN_MOVES_REQUIRED} moves are useless for training.\n"
                "Options:\n"
                "  1. Fix the source data (re-run selfplay with proper move recording)\n"
                "  2. Use --no-strict to export anyway (NOT RECOMMENDED - training will fail or produce poor models)\n"
            )
            return 1
        print(f"[STRICT MODE] All {len(db_paths)} database(s) passed strict validation\n")
    else:
        print("[WARNING] Running with --no-strict: Databases with incomplete move data may be included.")
        print("          This is NOT RECOMMENDED for production training.\n")

    _enforce_canonical_db_policy(
        db_paths,
        args.output,
        allow_noncanonical=bool(args.allow_noncanonical),
    )
    # Use central canonical source validation
    allowed_statuses = ["canonical", "pending_gate"] if args.allow_pending_gate else ["canonical"]
    enforce_canonical_sources(
        [Path(p) for p in db_paths],
        registry_path=Path(args.registry) if args.registry else None,
        allowed_statuses=allowed_statuses,
        allow_noncanonical=bool(args.allow_noncanonical),
        error_prefix="export-replay-dataset",
    )

    # Parse source filters (December 2025 - Phase 5 Unified NN/NNUE training)
    include_sources: set[str] | None = None
    exclude_sources: set[str] | None = None

    # Build include set from convenience flags and --sources
    if args.include_gauntlet or args.include_tournaments or args.include_human or args.sources:
        include_sources = set()
        include_sources.add("selfplay")  # Always include selfplay by default
        if args.include_gauntlet:
            include_sources.add("gauntlet")
            print("[SOURCE FILTER] Including gauntlet games")
        if args.include_tournaments:
            include_sources.add("tournament")
            print("[SOURCE FILTER] Including tournament games")
        if args.include_human:
            include_sources.add("human")
            include_sources.add("human_vs_ai")  # January 2026: human vs AI games from web client
            include_sources.add("sandbox")  # Sandbox may contain human games too
            print("[SOURCE FILTER] Including human gameplay (weighted 3x in training)")
        if args.sources:
            for source in args.sources.split(","):
                source = source.strip().lower()
                if source:
                    include_sources.add(source)
            print(f"[SOURCE FILTER] Sources: {', '.join(sorted(include_sources))}")

    # Build exclude set from --exclude-sources
    if args.exclude_sources:
        exclude_sources = set()
        for source in args.exclude_sources.split(","):
            source = source.strip().lower()
            if source:
                exclude_sources.add(source)
        print(f"[SOURCE FILTER] Excluding: {', '.join(sorted(exclude_sources))}")

    # Determine parallelism: default is parallel unless --single-threaded or --workers=1
    use_parallel = not args.single_threaded and (args.workers is None or args.workers > 1)

    # NOTE: As of Dec 2025, heuristics are now supported in parallel mode!
    # The parallel encoder extracts heuristics in worker processes (10-20x faster).

    # Use parallel export by default (10-20x faster on multi-core systems)
    if use_parallel:
        from scripts.export_replay_dataset_parallel import export_parallel
        num_workers = args.workers
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 4) - 1)
        print(f"[PARALLEL] Using {num_workers} worker processes for encoding")
        result = export_parallel(
            db_paths=db_paths,
            board_type=board_type,
            num_players=args.num_players,
            output_path=args.output,
            num_workers=num_workers,
            encoder_version=args.encoder_version,
            history_length=args.history_length,
            feature_version=args.feature_version,
            sample_every=args.sample_every,
            max_games=args.max_games,
            require_completed=args.require_completed,
            min_moves=args.min_moves,
            max_moves=args.max_moves,
            use_board_aware_encoding=args.board_aware_encoding,
            require_moves=not args.no_require_moves,
            use_cache=args.use_cache,
            force_export=args.force_export,
            include_heuristics=args.include_heuristics,
            full_heuristics=args.full_heuristics,
        )
        # December 2025 fix: result=0 means success, don't treat as falsy
        return result if result is not None else 1

    # Single-threaded mode (legacy, for debugging or when parallelism causes issues)
    print("[SINGLE-THREADED] Using sequential encoding (use --workers N for parallel mode)")

    # Check cache if enabled
    if args.use_cache:
        cache = get_export_cache()
        if not cache.needs_export(
            db_paths=db_paths,
            output_path=args.output,
            board_type=args.board_type,
            num_players=args.num_players,
            history_length=args.history_length,
            feature_version=args.feature_version,
            policy_encoding="board_aware" if args.board_aware_encoding else "legacy_max_n",
            force=args.force_export,
            # Quality filter params - critical for cache key uniqueness (Dec 28, 2025)
            min_quality=args.min_quality,
            require_completed=args.require_completed,
            encoder_version=args.encoder_version,
            include_heuristics=args.include_heuristics,
            full_heuristics=args.full_heuristics,
        ):
            cache_info = cache.get_cache_info(
                args.output,
                args.board_type,
                args.num_players,
                history_length=args.history_length,
                feature_version=args.feature_version,
                policy_encoding="board_aware" if args.board_aware_encoding else "legacy_max_n",
                # Quality filter params - critical for cache key uniqueness (Dec 28, 2025)
                min_quality=args.min_quality,
                require_completed=args.require_completed,
                encoder_version=args.encoder_version,
                include_heuristics=args.include_heuristics,
                full_heuristics=args.full_heuristics,
            )
            samples = cache_info.get("samples_exported", "?") if cache_info else "?"
            print("[CACHE HIT] Skipping export - source DBs unchanged since last export")
            print(f"  Output: {args.output}")
            print(f"  Cached samples: {samples}")
            return 0
        print("[CACHE MISS] Export needed - source DBs have changed")

    # Use multi-source export with deduplication
    export_replay_dataset_multi(
        db_paths=db_paths,
        board_type=board_type,
        num_players=args.num_players,
        output_path=args.output,
        history_length=args.history_length,
        feature_version=args.feature_version,
        sample_every=args.sample_every,
        max_games=args.max_games,
        require_completed=args.require_completed,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        max_move_index=args.max_move_index,
        use_rank_aware_values=not args.no_rank_aware_values,
        parity_fixtures_dir=args.parity_fixtures_dir,
        exclude_recovery=args.exclude_recovery,
        use_board_aware_encoding=args.board_aware_encoding,
        append=bool(args.append),
        encoder_version=args.encoder_version,
        require_moves=not args.no_require_moves,
        min_quality=args.min_quality,
        min_elo=args.min_elo,
        include_heuristics=args.include_heuristics,
        full_heuristics=args.full_heuristics,
        # Source filtering (December 2025 - Phase 5 Unified NN/NNUE training)
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        # Move data validation (January 2026 - Phase 6 Data Integrity)
        fail_on_orphans=args.strict,
        # Quality-weighted sampling
        quality_weighted=args.quality_weighted,
        # Quality tier filtering (January 2026 - Training Quality Fix)
        quality_tier=args.quality_tier,
    )

    # Update cache if enabled
    if args.use_cache:
        # Read sample count from output file
        samples_exported = 0
        games_exported = 0
        try:
            with np.load(args.output, allow_pickle=True) as data:
                if "values" in data:
                    samples_exported = len(data["values"])
        except (FileNotFoundError, PermissionError, OSError, ValueError, KeyError):
            pass

        cache.record_export(
            db_paths=db_paths,
            output_path=args.output,
            board_type=args.board_type,
            num_players=args.num_players,
            history_length=args.history_length,
            feature_version=args.feature_version,
            policy_encoding="board_aware" if args.board_aware_encoding else "legacy_max_n",
            samples_exported=samples_exported,
            games_exported=games_exported,
            # Quality filter params - critical for cache key uniqueness (Dec 28, 2025)
            min_quality=args.min_quality,
            require_completed=args.require_completed,
            encoder_version=args.encoder_version,
            include_heuristics=args.include_heuristics,
            full_heuristics=args.full_heuristics,
        )
        print(f"[CACHE] Recorded export: {samples_exported} samples")

    # Emit NPZ_EXPORT_COMPLETE event to trigger training
    try:
        import asyncio
        from app.coordination.event_emission_helpers import safe_emit_event_async

        # Read sample count from output file if not already done
        if not args.use_cache:
            samples_exported = 0
            try:
                with np.load(args.output, allow_pickle=True) as data:
                    if "values" in data:
                        samples_exported = len(data["values"])
            except (FileNotFoundError, PermissionError, OSError, ValueError, KeyError):
                pass

        async def _emit():
            await safe_emit_event_async(
                "NPZ_EXPORT_COMPLETE",
                {
                    "board_type": args.board_type,
                    "num_players": args.num_players,
                    "samples_exported": samples_exported,
                    "games_exported": games_exported if 'games_exported' in dir() else 0,
                    "output_path": args.output,
                    "success": True,
                    "feature_version": args.feature_version,
                    "encoder_version": args.encoder_version,
                },
                context="export_replay_dataset",
            )

        # Run async emission
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_emit())
        except RuntimeError:
            asyncio.run(_emit())

        print(f"[Event] Emitted NPZ_EXPORT_COMPLETE: {args.board_type}_{args.num_players}p, {samples_exported} samples")
    except ImportError:
        pass  # Event system not available
    except Exception as e:
        print(f"[Warning] Failed to emit export event: {e}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
