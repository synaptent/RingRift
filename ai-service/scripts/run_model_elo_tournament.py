#!/usr/bin/env python3
"""Run comprehensive model-vs-model tournament with persistent Elo tracking.

This script addresses the gap where 182+ trained models exist but no cross-model
Elo leaderboard tracks their relative strengths.

Features:
1. Discovers all trained models (.pth files)
2. Runs round-robin or Swiss tournaments between models
3. Persists Elo ratings to SQLite database
4. Generates leaderboard reports
5. Records canonical GameReplayDB replays with rich metadata (optional JSONL)

Usage:
    # Run tournament between all v3/v4/v5 models
    python scripts/run_model_elo_tournament.py --board square8 --players 2

    # Run quick tournament with top N models only
    python scripts/run_model_elo_tournament.py --board square8 --players 2 --top-n 10

    # View current leaderboard without running games
    python scripts/run_model_elo_tournament.py --leaderboard-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.ai.base import BaseAI

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Early startup heartbeat to prevent SSH timeout during slow imports
print("[startup] Loading tournament script...")
sys.stdout.flush()

# Import event bus helpers (consolidated imports)
import asyncio

print("[startup] Importing torch...")
sys.stdout.flush()
import torch
print("[startup] Importing app modules...")
sys.stdout.flush()

from app.distributed.event_helpers import (
    emit_elo_updated_safe,
    get_event_bus_safe,
    has_event_bus,
)
from app.db.unified_recording import (
    RecordingConfig,
    is_recording_enabled,
    record_game_unified,
)
from app.models import (
    AIConfig,
    AIType,
    BoardType,
    GameState,
    GameStatus,
)
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state
from app.utils.victory_type import derive_victory_type
from app.utils.parallel_defaults import get_tournament_workers
from app.config.thresholds import ARCHIVE_ELO_THRESHOLD
from scripts.lib.resilience import exponential_backoff_delay
from scripts.lib.tournament_cli import (
    archive_low_elo_models,
    filter_archived_models,
    generate_elo_based_matchups,
    is_model_archived,
    unarchive_discovered_models,
    unarchive_model,
)

# Composite participant support for per-algorithm ELO tracking
try:
    from app.training.composite_participant import (
        make_composite_participant_id,
        STANDARD_ALGORITHM_CONFIGS,
    )
    HAS_COMPOSITE = True
except ImportError:
    HAS_COMPOSITE = False
    make_composite_participant_id = None
    STANDARD_ALGORITHM_CONFIGS = {}

HAS_EVENT_BUS = has_event_bus()

# For backwards compatibility
if HAS_EVENT_BUS:
    from app.coordination.event_router import emit_elo_updated
else:
    emit_elo_updated = emit_elo_updated_safe

# Import coordination helpers (consolidated imports)
from app.coordination.helpers import (
    OrchestratorRole,
    can_spawn_safe,
    get_registry_safe,
    has_coordination,
)

HAS_COORDINATION = has_coordination()

# For backwards compatibility
if HAS_COORDINATION:
    from app.coordination import (
        can_schedule_task,
        can_spawn,
        estimate_task_duration,
        get_registry,
        record_task_completion,
        register_running_task,
    )
else:
    get_registry = get_registry_safe
    can_spawn = can_spawn_safe
    can_schedule_task = None
    estimate_task_duration = None
    register_running_task = None
    record_task_completion = None

print("[startup] Imports complete, ready to run")
sys.stdout.flush()

_RECORDING_LOCK = threading.Lock()


# ============================================
# Source Tag Configuration
# ============================================
# Default source tag - identifies games as tournament (filtered from training)
# When --training-mode is used, this is changed to "elo_selfplay" so games
# feed into training pool instead of being filtered as holdout
GAME_SOURCE_TAG = "run_model_elo_tournament"

# Timeout handling constants
DEFAULT_MATCHUP_TIMEOUT = 600  # 10 minutes per matchup
DEFAULT_TOURNAMENT_TIMEOUT = 7200  # 2 hours for entire tournament
HEARTBEAT_INTERVAL = 60  # Log progress every 60 seconds

# Validation result cache to avoid repeated checkpoint loading during discovery
# Key: (model_path, board_type, num_players, file_mtime)
# Value: (is_valid, reason)
_VALIDATION_CACHE: dict[tuple[str, str, int, float], tuple[bool, str]] = {}


# ============================================
# JSON Serialization Helpers
# ============================================

class GameRecordEncoder(json.JSONEncoder):
    """Custom JSON encoder for game records with non-serializable types."""

    def default(self, obj):
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Pydantic models
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict"):
            return obj.dict()
        # Handle enums
        if hasattr(obj, "value"):
            return obj.value
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return super().default(obj)


def serialize_game_state(state) -> dict[str, Any]:
    """Serialize a GameState to a JSON-compatible dict."""
    if hasattr(state, "model_dump"):
        return state.model_dump(mode="json")
    if hasattr(state, "dict"):
        # Convert the dict and handle any remaining non-serializable values
        raw = state.dict()
        return json.loads(json.dumps(raw, cls=GameRecordEncoder))
    return {}


def _build_recording_config(
    board_type: str,
    num_players: int,
    *,
    source_tag: str,
    db_dir: str,
    db_prefix: str,
    db_path: str | None,
    tags: list[str],
) -> RecordingConfig:
    return RecordingConfig(
        board_type=board_type,
        num_players=num_players,
        source=source_tag,
        engine_mode="model_elo_tournament",
        db_dir=db_dir,
        db_prefix=db_prefix,
        db_path=db_path,
        tags=tags,
        store_history_entries=True,
    )


def _recording_tags_for_args(args: argparse.Namespace, board_type: str, num_players: int) -> list[str]:
    tags = [
        "elo_tournament",
        f"board_{board_type}",
        f"players_{num_players}",
    ]
    if getattr(args, "training_mode", False):
        tags.append("training_mode")
    if getattr(args, "elo_matchmaking", False):
        tags.append("elo_matchmaking")
    if getattr(args, "both_ai_types", False):
        tags.append("multi_ai_types")
    if getattr(args, "baselines_only", False):
        tags.append("baselines_only")
    return tags


def _matchup_id_for_models(model_a_id: str, model_b_id: str) -> str:
    digest = hashlib.sha1(f"{model_a_id}__{model_b_id}".encode("utf-8")).hexdigest()
    return digest[:12]


def _record_game_if_enabled(
    *,
    recording_config: RecordingConfig | None,
    initial_state: GameState | None,
    final_state: GameState | None,
    moves: list[Any] | None,
    game_id: str | None,
    extra_metadata: dict[str, Any],
    lock: threading.Lock | None,
) -> None:
    if recording_config is None:
        return
    if initial_state is None or final_state is None or moves is None:
        return
    try:
        if lock is None:
            record_game_unified(
                config=recording_config,
                initial_state=initial_state,
                final_state=final_state,
                moves=moves,
                extra_metadata=extra_metadata,
                game_id=game_id,
            )
        else:
            with lock:
                record_game_unified(
                    config=recording_config,
                    initial_state=initial_state,
                    final_state=final_state,
                    moves=moves,
                    extra_metadata=extra_metadata,
                    game_id=game_id,
                )
    except Exception as e:
        print(f"[Tournament] Recording failed for game {game_id}: {e}")


def _register_composite_participant(
    db: "EloDatabase",
    participant_id: str,
    model: dict[str, Any],
    ai_type: str,
    config: dict[str, Any],
) -> None:
    """Register a composite participant with metadata for ELO tracking."""
    metadata = {
        "is_composite": True,
        "parent_participant_id": model.get("model_id"),
        "nn_model_id": model.get("model_id"),
        "nn_model_path": model.get("model_path"),
        "ai_type": ai_type,
        "algorithm_config": config,
    }
    db.register_participant(
        participant_id=participant_id,
        participant_type="model",
        ai_type=ai_type,
        use_neural_net=True,
        model_path=model.get("model_path"),
        model_version=model.get("version"),
        metadata=metadata,
    )


# ============================================
# Model Validation
# ============================================

def validate_model_for_board(
    model_path: str | Path,
    board_type: BoardType,
    num_players: int = 2,
    run_inference_probe: bool = False,
) -> tuple[bool, str]:
    """Validate that a model checkpoint is compatible with the given board type.

    Args:
        model_path: Path to the model checkpoint
        board_type: Expected board type
        num_players: Expected player count
        run_inference_probe: If True, run actual inference to validate (slower but definitive)

    Returns:
        (is_valid, reason) - True if compatible, False with reason if not.
    """
    from app.utils.torch_utils import safe_load_checkpoint

    path = Path(model_path)
    if not path.exists():
        return False, f"Model file not found: {path}"

    # Skip validation for baseline players
    if str(model_path).startswith("__BASELINE_"):
        return True, "baseline"

    # Check validation cache (keyed by path, board_type, num_players, file_mtime)
    try:
        file_mtime = path.stat().st_mtime
    except OSError:
        file_mtime = 0.0

    board_key = str(board_type.value if hasattr(board_type, 'value') else board_type)
    cache_key = (str(path), board_key, num_players, file_mtime)
    if cache_key in _VALIDATION_CACHE:
        return _VALIDATION_CACHE[cache_key]

    def cache_and_return(result: tuple[bool, str]) -> tuple[bool, str]:
        """Cache validation result before returning."""
        _VALIDATION_CACHE[cache_key] = result
        return result

    try:
        checkpoint = safe_load_checkpoint(str(path), map_location="cpu", warn_on_unsafe=False)
    except Exception as e:
        return cache_and_return((False, f"Failed to load checkpoint: {e}"))

    # Expected policy sizes for each board type
    POLICY_SIZES = {
        "square8": 7000,
        "square19": 67000,
        "hex8": 4500,
        "hexagonal": 91876,
    }

    # Expected board sizes (spatial dimension)
    BOARD_SIZES = {
        "square8": 8,
        "square19": 19,
        "hex8": 9,       # 9x9 bounding box for radius-4 hex
        "hexagonal": 25,  # 25x25 bounding box for radius-12 hex
    }

    expected_board = str(board_type.value if hasattr(board_type, 'value') else board_type).lower()
    expected_policy_size = POLICY_SIZES.get(expected_board, 0)
    expected_board_size = BOARD_SIZES.get(expected_board, 0)

    # Check versioning metadata first (most reliable)
    versioning_metadata = checkpoint.get("_versioning_metadata", {})
    if versioning_metadata:
        config = versioning_metadata.get("config", {})
        model_board_type = config.get("board_type", "")
        model_board_size = config.get("board_size", 0)
        model_policy_size = config.get("policy_size", 0)

        if model_board_type:
            actual = str(model_board_type).lower()
            # Strict matching for hex8 vs hexagonal (they are NOT compatible)
            if expected_board in ["hex8", "hexagonal"] and actual in ["hex8", "hexagonal"]:
                if expected_board != actual:
                    return cache_and_return((False, f"Hex board size mismatch: model is {actual}, expected {expected_board}"))
            elif expected_board in ["square8", "sq8"] and actual in ["square8", "sq8"]:
                pass  # Compatible
            elif expected_board in ["square19", "sq19"] and actual in ["square19", "sq19"]:
                pass  # Compatible
            elif expected_board != actual:
                return cache_and_return((False, f"Board type mismatch: model is {actual}, expected {expected_board}"))

        # Board size is the most reliable indicator
        if model_board_size and expected_board_size and model_board_size != expected_board_size:
            return cache_and_return((False, f"Board size mismatch: model has {model_board_size}, expected {expected_board_size}"))

        # Policy size can vary based on action encoding, so only warn if very different
        # (disabled for now since policy sizes vary widely)

    # Check old-style metadata
    metadata = checkpoint.get("metadata", {})
    if metadata:
        model_board_type = metadata.get("board_type", "")
        if model_board_type:
            actual = str(model_board_type).lower()
            if expected_board in ["hex8", "hexagonal"] and actual in ["hex8", "hexagonal"]:
                if expected_board != actual:
                    return cache_and_return((False, f"Hex board size mismatch: model is {actual}, expected {expected_board}"))

    # Infer from architecture if metadata missing
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    if isinstance(state_dict, dict):
        # Check hex_mask shape (definitive for hex models)
        if "hex_mask" in state_dict:
            hex_mask_shape = state_dict["hex_mask"].shape
            # hex_mask is (1, 1, board_size, board_size)
            if len(hex_mask_shape) == 4:
                mask_size = hex_mask_shape[2]
                if expected_board == "hex8" and mask_size != 9:
                    return cache_and_return((False, f"hex_mask size {mask_size} doesn't match hex8 (expected 9)"))
                elif expected_board == "hexagonal" and mask_size != 25:
                    return cache_and_return((False, f"hex_mask size {mask_size} doesn't match hexagonal (expected 25)"))
                elif expected_board in ["square8", "square19"]:
                    return cache_and_return((False, f"Model has hex_mask but expected square board type"))

        # Check board_size from state_dict keys if hex_mask wasn't found
        # This is less reliable but still useful
        if "hex_mask" not in state_dict:
            # For square boards, check if any hex-specific keys exist
            hex_keys = ["hex_mask", "hex_conv", "axial_conv"]
            has_hex_architecture = any(k for k in state_dict.keys() if any(hk in k.lower() for hk in hex_keys))

            if has_hex_architecture and expected_board in ["square8", "square19"]:
                return cache_and_return((False, f"Model has hex architecture but expected square board"))
            elif not has_hex_architecture and expected_board in ["hex8", "hexagonal"]:
                return cache_and_return((False, f"Model lacks hex architecture but expected hex board"))

    # Optional: Run actual inference probe (slowest but most reliable)
    if run_inference_probe:
        try:
            probe_result = _run_inference_probe(str(path), board_type, num_players)
            if not probe_result[0]:
                return cache_and_return(probe_result)
        except Exception as e:
            return cache_and_return((False, f"Inference probe failed: {e}"))

    return cache_and_return((True, "compatible"))


def _run_inference_probe(model_path: str, board_type: BoardType, num_players: int) -> tuple[bool, str]:
    """Run a single inference pass to verify model compatibility."""
    try:
        from app.ai.neural_net import NeuralNetAI
        from app.models import AIConfig, AIType
        from app.training.initial_state import create_initial_state

        config = AIConfig(
            ai_type=AIType.NEURAL_DEMO,
            board_type=board_type,
            nn_model_id=model_path,
        )
        nn_ai = NeuralNetAI(player_number=1, config=config, board_type=board_type)

        # Create test state and run inference
        test_state = create_initial_state(board_type, num_players)
        values, policy = nn_ai.evaluate_batch([test_state])

        if len(values) != 1:
            return False, f"Inference returned {len(values)} values, expected 1"

        return True, "inference_ok"
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            return False, f"Architecture mismatch during inference: {error_msg[:200]}"
        return False, f"Inference failed: {error_msg[:200]}"


# ============================================
# Game Execution with Neural Networks
# ============================================

def create_ai_from_model(
    model_def: dict[str, Any],
    player_number: int,
    board_type: BoardType,
    game_seed: int = 0,
) -> BaseAI:
    """Create an AI instance from a model definition.

    Supports neural network models (model_path to .pth file), NNUE models (.pt files),
    and baseline players (model_path starting with __BASELINE_).

    Args:
        game_seed: Per-game seed for varied randomness (Jan 2026 fix).
                   Without this, RandomAI/HeuristicAI produce identical games.
    """
    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.mcts_ai import MCTSAI
    from app.ai.minimax_ai import MinimaxAI
    from app.ai.random_ai import RandomAI

    model_path = model_def.get("model_path", "")
    ai_type = model_def.get("ai_type", "neural_net")

    # Per-game seed for varied randomness
    rng_seed = (game_seed * 104729 + player_number * 7919) & 0xFFFFFFFF

    # Check if model file exists (skip for baseline players)
    if model_path and not model_path.startswith("__BASELINE_") and not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run cleanup to remove stale DB entries."
        )

    if model_path == "__BASELINE_RANDOM__" or ai_type == "random":
        config = AIConfig(ai_type=AIType.RANDOM, board_type=board_type, difficulty=1, rng_seed=rng_seed)
        return RandomAI(player_number, config)

    elif model_path == "__BASELINE_HEURISTIC__" or ai_type == "heuristic":
        config = AIConfig(ai_type=AIType.HEURISTIC, board_type=board_type, difficulty=5, rng_seed=rng_seed)
        return HeuristicAI(player_number, config)

    elif model_path.startswith("__BASELINE_MCTS") or ai_type == "mcts":
        mcts_sims = model_def.get("mcts_simulations", 100)
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=7,
            mcts_iterations=mcts_sims,
            use_neural_net=False,  # Pure heuristic MCTS for baselines
        )
        return MCTSAI(player_number, config)

    elif ai_type == "nnue" or model_path.endswith(".pt"):
        # NNUE model - use MinimaxAI with NNUE evaluation
        config = AIConfig(
            ai_type=AIType.MINIMAX,
            board_type=board_type,
            difficulty=5,  # D5 uses NNUE evaluation
            use_neural_net=True,  # Enable NNUE
        )
        return MinimaxAI(player_number, config)

    elif ai_type == "gmo" or model_path == "__BASELINE_GMO__":
        # GMO (Gradient Move Optimization) model
        from app.ai.gmo_ai import GMOAI, GMOConfig
        gmo_config = GMOConfig(device="cpu")
        config = AIConfig(ai_type=AIType.GMO, board_type=board_type, difficulty=5)
        gmo_ai = GMOAI(player_number, config, gmo_config)
        # Load checkpoint if specified
        if model_path and not model_path.startswith("__BASELINE_"):
            gmo_ai.load_checkpoint(Path(model_path))
        return gmo_ai

    elif ai_type == "ebmo" or model_path == "__BASELINE_EBMO__":
        # EBMO (Energy-Based Move Optimization) model
        from app.ai.ebmo_ai import EBMO_AI
        from app.ai.ebmo_network import EBMOConfig
        # Use fast config for tournament (direct eval instead of gradient descent)
        ebmo_config = EBMOConfig(
            use_direct_eval=True,  # Skip gradient descent for speed
            board_size=8 if board_type == BoardType.SQUARE8 else 19,
        )
        config = AIConfig(ai_type=AIType.EBMO, board_type=board_type, difficulty=5)
        ebmo_model_path = model_path if model_path and not model_path.startswith("__BASELINE_") else None
        return EBMO_AI(player_number, config, ebmo_model_path, ebmo_config)

    elif ai_type == "gmo_mcts" or model_path == "__BASELINE_GMO_MCTS__":
        # GMO-MCTS Hybrid - combines GMO gradient scoring with MCTS tree search
        from app.ai.gmo_mcts_hybrid import GMOMCTSConfig, GMOMCTSHybrid
        num_sims = model_def.get("mcts_simulations", 100)
        hybrid_config = GMOMCTSConfig(
            num_simulations=num_sims,
            c_puct=1.5,
            use_gmo_prior=True,
            gmo_prior_weight=0.7,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        config = AIConfig(ai_type=AIType.GMO_MCTS, board_type=board_type, difficulty=7)
        return GMOMCTSHybrid(player_number, config, hybrid_config)

    elif ai_type == "ig_gmo" or model_path == "__BASELINE_IG_GMO__":
        # Information-Gain GMO with MI-based exploration and GNN encoder
        from app.ai.ig_gmo import IGGMO, IGGMOConfig
        ig_config = IGGMOConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gnn=True,
            mi_exploration_weight=0.3,
        )
        config = AIConfig(ai_type=AIType.IG_GMO, board_type=board_type, difficulty=7)
        return IGGMO(player_number, config, ig_config)

    else:
        # Neural network model - use MCTS with neural net guidance
        # Extract model_id from model_path (strip .pth extension)
        model_id = Path(model_path).stem if model_path else None
        mcts_iters = model_def.get("mcts_iterations", 800)
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            model_id=model_id,
            mcts_iterations=mcts_iters,
        )
        return MCTSAI(player_number, config)


def play_model_vs_model_game(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 10000,
    save_game_history: bool = True,
    game_seed: int = 0,
) -> dict[str, Any]:
    """Play a single game between two models (NN or baseline).

    Returns dict with: winner (model_a, model_b, or draw), game_length, duration_sec, game_record
    If save_game_history=True, also returns full game record for training data export.

    Args:
        game_seed: Per-game seed for varied randomness (Jan 2026 fix).
                   Without this, RandomAI/HeuristicAI produce identical games.
    """
    import uuid

    from app.rules.default_engine import DefaultRulesEngine
    from app.training.initial_state import create_initial_state

    start_time = time.time()
    game_id = str(uuid.uuid4())

    # Create initial state
    state = create_initial_state(board_type, num_players)
    state.id = game_id
    initial_state = state.copy(deep=True)  # Pydantic v1 compatibility
    engine = DefaultRulesEngine()

    # Capture initial state for training data (use JSON-safe serialization)
    initial_state_snapshot = serialize_game_state(state) if save_game_history else None
    move_history = []
    moves_played = []
    termination_reason = None

    # Create AIs for both models (pass game_seed for unique randomness - Jan 2026 fix)
    # Model A plays as player 1, Model B plays as player 2
    ai_a = create_ai_from_model(model_a, 1, board_type, game_seed)
    ai_b = create_ai_from_model(model_b, 2, board_type, game_seed)

    # Check if neural net models actually loaded their networks
    # If they fell back to heuristic, we should not record under NN name
    fallback_used = False
    fallback_models = []

    for ai, model in [(ai_a, model_a), (ai_b, model_b)]:
        model_path = model.get("model_path", "")
        ai_type = model.get("ai_type", "")

        # Only check NN models (not baselines)
        if not model_path.startswith("__BASELINE") and ai_type not in ("random", "heuristic", "mcts"):
            # Check if neural_net attribute exists and is None (fallback occurred)
            nn = getattr(ai, "neural_net", "NOT_FOUND")
            if nn is None:
                fallback_used = True
                fallback_models.append(model.get("model_id", "unknown"))
                print(f"  [WARNING] Model {model.get('model_id')} fell back to heuristic (NN failed to load)")

    if fallback_used:
        return {
            "winner": "skipped",
            "game_length": 0,
            "duration_sec": 0,
            "game_id": game_id,
            "error": f"Neural net fallback detected for: {fallback_models}",
            "termination_reason": "nn_fallback",
            "recordable": False,
            "fallback_models": fallback_models,
        }

    # Track player types for metadata
    player_types = []
    for m in [model_a, model_b]:
        ai_type = m.get("ai_type", "")
        if ai_type in ("random", "heuristic", "mcts"):
            player_types.append(ai_type)
        elif m.get("model_path", "").startswith("__BASELINE"):
            player_types.append("baseline")
        else:
            player_types.append("neural_net")

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player
        ai = ai_a if current_player == 1 else ai_b

        # Select move (BaseAI.select_move handles getting valid moves internally)
        move = ai.select_move(state)
        if move is None:
            fallback_moves = engine.get_valid_moves(state, current_player)
            if fallback_moves:
                move = fallback_moves[0]
            else:
                termination_reason = "no_valid_moves"
                break

        # Record move for training data
        # Move object uses 'type' attribute (not 'move_type') - match play_nn_vs_nn_game format
        if save_game_history and move is not None:
            move_type_val = move.type.value if hasattr(move.type, 'value') else str(move.type)
            move_record = {
                'type': move_type_val,
                'player': current_player,
            }
            if hasattr(move, 'to_key') and move.to_key:
                move_record['to_key'] = move.to_key
            if hasattr(move, 'to') and move.to:
                move_record['to'] = {'x': move.to.x, 'y': move.to.y}
            if hasattr(move, 'from_key') and move.from_key:
                move_record['from_key'] = move.from_key
            if hasattr(move, 'from_pos') and move.from_pos:
                move_record['from'] = {'x': move.from_pos.x, 'y': move.from_pos.y}
            if hasattr(move, 'ring_index') and move.ring_index is not None:
                move_record['ring_index'] = move.ring_index
            move_history.append(move_record)

        moves_played.append(move)

        # Apply move
        try:
            state = engine.apply_move(state, move, trace_mode=True)
        except Exception as e:
            return {
                "winner": "error",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "game_id": game_id,
                "error": f"Move error: {e}",
                "initial_state": initial_state,
                "final_state": state,
                "moves": moves_played,
                "termination_reason": "apply_move_error",
                "recordable": False,
            }
        move_count += 1

    duration = time.time() - start_time

    # Determine winner
    winner = "draw"
    winner_player = None
    if state.game_status == GameStatus.COMPLETED:
        if state.winner == 1:
            winner = "model_a"
            winner_player = 1
        elif state.winner == 2:
            winner = "model_b"
            winner_player = 2

    status = "completed" if state.game_status == GameStatus.COMPLETED else str(state.game_status.value)
    if termination_reason is None:
        if move_count >= max_moves and state.game_status == GameStatus.ACTIVE:
            termination_reason = "max_moves"
        else:
            termination_reason = status

    # Build game record for training data export
    game_record = None
    if save_game_history:
        game_record = {
            'game_id': game_id,
            'board_type': board_type.value if hasattr(board_type, 'value') else str(board_type),
            'num_players': num_players,
            'winner': winner_player,
            'move_count': move_count,
            'total_moves': move_count,
            'status': status,
            'game_status': status,
            'termination_reason': termination_reason,
            'completed': state.game_status == GameStatus.COMPLETED,
            'engine_mode': 'mixed_tournament',
            'opponent_type': 'tournament_baseline',
            'player_types': player_types,
            'model_a': model_a.get("model_id", model_a.get("model_path", "unknown")),
            'model_b': model_b.get("model_id", model_b.get("model_path", "unknown")),
            'moves': move_history,
            'initial_state': initial_state_snapshot,
            'game_time_seconds': duration,
            'duration_sec': duration,
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),
            'source': GAME_SOURCE_TAG,
        }

    return {
        "winner": winner,
        "game_length": move_count,
        "duration_sec": duration,
        "game_id": game_id,
        "initial_state": initial_state,
        "final_state": state,
        "moves": moves_played,
        "termination_reason": termination_reason,
        "recordable": termination_reason != "no_valid_moves",
        "final_status": state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status),
        "game_record": game_record,
    }


def play_nn_vs_nn_game(
    model_a_path: str,
    model_b_path: str,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 10000,
    mcts_simulations: int = 100,
    save_game_history: bool = True,
    ai_type: str = "descent",
    ai_type_a: str | None = None,
    ai_type_b: str | None = None,
) -> dict[str, Any]:
    """Play a single game between two neural network models.

    Returns dict with: winner (model_a, model_b, or draw), game_length, duration_sec, game_record
    If save_game_history=True, also returns full game record for training data export.
    The game_record follows canonical JSONL format suitable for NPZ conversion.

    If ai_type_a and ai_type_b are provided, they are used for model A and B respectively,
    enabling cross-inference matches (e.g., model_a uses MCTS, model_b uses Descent).
    Otherwise, ai_type is used for both models.
    """
    import time
    import uuid
    from datetime import datetime

    from app.ai.neural_net import NeuralNetAI, clear_model_cache
    from app.ai.model_cache import set_tournament_mode

    def _timeout_tiebreak_winner(final_state: GameState) -> int | None:
        """Deterministically select a winner for evaluation-only timeouts."""
        players = getattr(final_state, "players", None) or []
        if not players:
            return None

        territory_counts: dict[int, int] = {}
        try:
            for p_id in final_state.board.collapsed_spaces.values():
                territory_counts[int(p_id)] = territory_counts.get(int(p_id), 0) + 1
        except (AttributeError, TypeError, ValueError, KeyError):
            pass

        marker_counts: dict[int, int] = {int(p.player_number): 0 for p in players}
        try:
            for marker in final_state.board.markers.values():
                owner = int(marker.player)
                marker_counts[owner] = marker_counts.get(owner, 0) + 1
        except (AttributeError, TypeError, ValueError, KeyError):
            pass

        last_actor: int | None = None
        try:
            if final_state.move_history:
                last_actor = int(final_state.move_history[-1].player)
        except (AttributeError, TypeError, ValueError, IndexError):
            last_actor = None

        sorted_players = sorted(
            players,
            key=lambda p: (
                territory_counts.get(int(p.player_number), 0),
                int(getattr(p, "eliminated_rings", 0) or 0),
                marker_counts.get(int(p.player_number), 0),
                1 if last_actor == int(p.player_number) else 0,
                -int(p.player_number),
            ),
            reverse=True,
        )
        if not sorted_players:
            return None
        return int(sorted_players[0].player_number)

    def _winner_label_for_player(player_num: int) -> str:
        # Players are assigned model_a/model_b alternating by position index:
        # P1,P3 -> model_a; P2,P4 -> model_b.
        return "model_a" if ((int(player_num) - 1) % 2) == 0 else "model_b"

    # Move history for training data export
    move_history = []
    moves_played = []
    termination_reason = None

    start_time = time.time()

    # Use canonical create_initial_state for proper setup
    game_state = create_initial_state(board_type=board_type, num_players=num_players)
    game_state.id = str(uuid.uuid4())
    initial_state = game_state.copy(deep=True)  # Pydantic v1

    # Capture initial state snapshot for NPZ export (required for training data)
    # Use serialize_game_state() for JSON-safe serialization
    initial_state_snapshot = serialize_game_state(game_state) if save_game_history else None

    # Create AI instances - alternate between model A and model B
    # Player 1 -> model_a, Player 2 -> model_b
    ai_configs = []
    model_paths = [model_a_path, model_b_path]

    # Determine AI types for each player
    # Use per-player types if provided, otherwise use the shared ai_type
    effective_ai_type_a = ai_type_a if ai_type_a else ai_type
    effective_ai_type_b = ai_type_b if ai_type_b else ai_type
    ai_types = [effective_ai_type_a, effective_ai_type_b]

    # Map AI type strings to AIType enum and create configs
    # All 11 AI types for diverse high-quality training data
    AI_TYPE_MAP = {
        "random": AIType.RANDOM,
        "heuristic": AIType.HEURISTIC,
        "minimax": AIType.MINIMAX,
        "gpu_minimax": AIType.GPU_MINIMAX,  # GPU batched minimax
        "mcts": AIType.MCTS,
        "descent": AIType.DESCENT,
        "policy_only": AIType.POLICY_ONLY,
        "gumbel_mcts": AIType.GUMBEL_MCTS,
        "maxn": AIType.MAXN,           # Multiplayer: each player maximizes own score
        "brs": AIType.BRS,             # Multiplayer: best-reply search (fast)
        "neural_demo": AIType.NEURAL_DEMO,  # Experimental neural
    }

    for i in range(num_players):
        model_idx = i % 2  # Alternate models for multiplayer
        player_ai_type = ai_types[model_idx]
        ai_type_enum = AI_TYPE_MAP.get(player_ai_type, AIType.DESCENT)
        config = AIConfig(
            type=ai_type_enum,
            difficulty=10,
            nn_model_id=model_paths[model_idx],  # Pass full path
            mcts_simulations=mcts_simulations,
            think_time=5000,
            use_neural_net=True,
        )
        ai_configs.append(config)

    # Create AIs based on their type
    ais = []
    try:
        for i, config in enumerate(ai_configs):
            player_ai_type = ai_types[i % 2]
            if player_ai_type == "policy_only":
                from app.ai.policy_only_ai import PolicyOnlyAI
                ai = PolicyOnlyAI(player_number=i + 1, config=config, board_type=board_type)
            elif player_ai_type == "gumbel_mcts":
                from app.ai.gumbel_mcts_ai import GumbelMCTSAI
                ai = GumbelMCTSAI(player_number=i + 1, config=config, board_type=board_type)
            elif player_ai_type == "maxn":
                from app.ai.maxn_ai import MaxNAI
                ai = MaxNAI(player_number=i + 1, config=config)
            elif player_ai_type == "brs":
                from app.ai.maxn_ai import BRSAI
                ai = BRSAI(player_number=i + 1, config=config)
            else:
                ai = NeuralNetAI(player_number=i + 1, config=config, board_type=board_type)
            ais.append(ai)
    except Exception as e:
        clear_model_cache()
        return {
            "winner": "error",
            "game_length": 0,
            "duration_sec": time.time() - start_time,
            "error": str(e),
        }

    # Check if any neural net model fell back to heuristic
    fallback_used = False
    fallback_models = []
    for i, ai in enumerate(ais):
        model_path = model_paths[i % 2]
        # Check if neural_net attribute exists and is None (fallback occurred)
        nn = getattr(ai, "neural_net", "NOT_FOUND")
        if nn is None:
            fallback_used = True
            model_name = Path(model_path).stem if model_path else f"player_{i+1}"
            fallback_models.append(model_name)
            print(f"  [WARNING] Model {model_name} fell back to heuristic (NN failed to load)")

    if fallback_used:
        clear_model_cache()
        return {
            "winner": "skipped",
            "game_length": 0,
            "duration_sec": time.time() - start_time,
            "game_id": game_state.id,
            "error": f"Neural net fallback detected for: {fallback_models}",
            "termination_reason": "nn_fallback",
            "recordable": False,
            "fallback_models": fallback_models,
        }

    rules_engine = DefaultRulesEngine()
    move_count = 0

    # Play the game
    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = game_state.current_player
        current_ai = ais[current_player - 1]
        current_ai.player_number = current_player

        try:
            move = current_ai.select_move(game_state)
        except Exception as e:
            # AI error - opponent wins
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "error": f"AI error: {e}",
            }

        if not move:
            fallback_moves = rules_engine.get_valid_moves(game_state, current_player)
            if fallback_moves:
                move = fallback_moves[0]
            else:
                # No valid moves - opponent wins (not recordable for canonical DBs)
                winner_idx = 1 if current_player == 1 else 0
                clear_model_cache()
                return {
                    "winner": "model_b" if winner_idx == 1 else "model_a",
                    "game_length": move_count,
                    "duration_sec": time.time() - start_time,
                    "game_id": game_state.id,
                    "error": "no_valid_moves",
                    "initial_state": initial_state,
                    "final_state": game_state,
                    "moves": moves_played,
                    "termination_reason": "no_valid_moves",
                    "recordable": False,
                }

        # Record move for training data in canonical format (matching run_random_selfplay.py)
        if save_game_history:
            # Move object uses 'type' attribute (not 'move_type')
            move_type_val = move.type.value if hasattr(move.type, 'value') else str(move.type)
            move_record = {
                'type': move_type_val,
                'player': current_player,
            }
            # Add position data (handle both key-based and coordinate-based moves)
            if hasattr(move, 'to_key') and move.to_key:
                move_record['to_key'] = move.to_key
            if hasattr(move, 'to') and move.to:
                move_record['to'] = {'x': move.to.x, 'y': move.to.y}
            if hasattr(move, 'from_key') and move.from_key:
                move_record['from_key'] = move.from_key
            if hasattr(move, 'from_pos') and move.from_pos:
                move_record['from'] = {'x': move.from_pos.x, 'y': move.from_pos.y}
            if hasattr(move, 'ring_index') and move.ring_index is not None:
                move_record['ring_index'] = move.ring_index
            move_history.append(move_record)

        moves_played.append(move)

        try:
            game_state = rules_engine.apply_move(game_state, move, trace_mode=True)
        except Exception as e:
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "game_id": game_state.id,
                "error": f"Move error: {e}",
                "initial_state": initial_state,
                "final_state": game_state,
                "moves": moves_played,
                "termination_reason": "apply_move_error",
                "recordable": False,
            }

        move_count += 1

    # Determine winner
    duration = time.time() - start_time
    # NOTE: Model cache is preserved for tournament efficiency.
    # Cache clearing after every game caused 74+ min stuck on model loading.
    # LRU eviction handles memory; cache is cleared at tournament end.

    # Derive victory type for canonical format
    victory_type, stalemate_tb = derive_victory_type(game_state, max_moves)
    status = "completed" if game_state.game_status == GameStatus.COMPLETED else str(game_state.game_status.value)
    if termination_reason is None:
        if move_count >= max_moves and game_state.game_status == GameStatus.ACTIVE:
            termination_reason = "max_moves"
        else:
            termination_reason = status

    # Evaluation-only timeout tie-break (avoid draw-heavy tournaments).
    winner_player: int | None = None
    if game_state.winner is not None:
        try:
            winner_player = int(game_state.winner)
        except (ValueError, TypeError):
            winner_player = None

    timed_out = bool(move_count >= max_moves and winner_player is None)
    evaluation_tiebreak_player: int | None = None
    if winner_player is None:
        evaluation_tiebreak_player = _timeout_tiebreak_winner(game_state)

    # Build game record for training data export in canonical format (matching run_random_selfplay.py)
    game_record = None
    if save_game_history:
        game_record = {
            # === Core game identifiers ===
            'game_id': game_state.id,
            'board_type': board_type.value if hasattr(board_type, 'value') else str(board_type),
            'num_players': num_players,
            # === Game outcome ===
            'winner': game_state.winner if game_state.game_status == GameStatus.COMPLETED else None,
            'move_count': move_count,
            'total_moves': move_count,  # Alias for compatibility
            'status': status,
            'game_status': status,
            'victory_type': victory_type,
            'stalemate_tiebreaker': stalemate_tb,
            'termination_reason': termination_reason,
            'completed': game_state.game_status == GameStatus.COMPLETED,
            # === Engine/opponent metadata ===
            'engine_mode': 'nn_vs_nn_tournament',
            'opponent_type': 'nn_tournament',
            'player_types': ['neural_net'] * num_players,
            'model_a': model_a_path,
            'model_b': model_b_path,
            # === Training data (required for NPZ export) ===
            'moves': move_history,
            'initial_state': initial_state_snapshot,
            # === Timing metadata ===
            'game_time_seconds': duration,
            'duration_sec': duration,
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),
            # === Source tracking ===
            'source': GAME_SOURCE_TAG,
        }
        if evaluation_tiebreak_player is not None:
            if timed_out:
                game_record["timeout_tiebreak_winner"] = int(evaluation_tiebreak_player)
                game_record["timeout_tiebreak_winner_model"] = _winner_label_for_player(int(evaluation_tiebreak_player))
            else:
                game_record["evaluation_tiebreak_winner"] = int(evaluation_tiebreak_player)
                game_record["evaluation_tiebreak_winner_model"] = _winner_label_for_player(int(evaluation_tiebreak_player))

    winner_label = "draw"
    if winner_player is not None:
        winner_label = _winner_label_for_player(winner_player)
    elif evaluation_tiebreak_player is not None:
        winner_label = _winner_label_for_player(evaluation_tiebreak_player)

    return {
        "winner": winner_label,
        "game_length": move_count,
        "duration_sec": duration,
        "game_id": game_state.id,
        "initial_state": initial_state,
        "final_state": game_state,
        "moves": moves_played,
        "termination_reason": termination_reason,
        "recordable": True,
        "game_record": game_record,
    }


def run_model_matchup(
    db: EloDatabase,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    board_type: str,
    num_players: int,
    games: int,
    tournament_id: str,
    nn_ai_type: str = "descent",
    use_both_ai_types: bool = False,
    save_games_dir: Path | None = None,
    jsonl_enabled: bool = True,
    recording_config: RecordingConfig | None = None,
    recording_metadata_base: dict[str, Any] | None = None,
    recording_lock: threading.Lock | None = None,
    use_composite: bool = False,
    game_retries: int = 3,
) -> dict[str, int]:
    """Run multiple games between two models and update Elo.

    If save_games_dir is provided, games are saved to JSONL for training data.
    If use_both_ai_types is True, runs half games with MCTS and half with Descent,
    ensuring neural networks are evaluated using both inference methods.
    """
    board_type_enum = BoardType.SQUARE8
    if board_type == "square19":
        board_type_enum = BoardType.SQUARE19
    elif board_type == "hex" or board_type == "hexagonal":
        board_type_enum = BoardType.HEXAGONAL
    elif board_type == "hex8":
        board_type_enum = BoardType.HEX8

    results = {"model_a_wins": 0, "model_b_wins": 0, "draws": 0, "errors": 0}

    # Setup game saving directory
    jsonl_path = None
    if jsonl_enabled:
        if save_games_dir is None:
            save_games_dir = AI_SERVICE_ROOT / "data" / "holdouts" / "elo_tournaments"
        save_games_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = save_games_dir / f"tournament_{tournament_id}_{board_type}_{num_players}p.jsonl"

    # Check if either model is a baseline (requires play_model_vs_model_game)
    is_baseline_match = (
        model_a.get("model_path", "").startswith("__BASELINE") or
        model_b.get("model_path", "").startswith("__BASELINE") or
        model_a.get("ai_type") in ("random", "heuristic", "mcts") or
        model_b.get("ai_type") in ("random", "heuristic", "mcts")
    )
    matchup_id = _matchup_id_for_models(model_a["model_id"], model_b["model_id"])
    recording_base = dict(recording_metadata_base or {})

    for game_num in range(games):
        # Alternate who plays first
        if game_num % 2 == 0:
            play_a, play_b = model_a, model_b
            id_a, id_b = model_a["model_id"], model_b["model_id"]
        else:
            play_a, play_b = model_b, model_a
            id_a, id_b = model_b["model_id"], model_a["model_id"]

        # Select AI type(s) for this game
        if use_both_ai_types:
            # Cross-inference evaluation: cycle through AI type combinations
            # This ensures each NN is evaluated with all inference methods
            if num_players == 2:
                # 2-player: Standard AI types
                ai_type_combos = [
                    ("descent", "descent"),         # Both use descent
                    ("mcts", "mcts"),               # Both use MCTS
                    ("policy_only", "policy_only"), # Both use Policy-Only
                    ("gumbel_mcts", "gumbel_mcts"), # Both use Gumbel MCTS
                    ("mcts", "descent"),            # Cross: MCTS vs Descent
                    ("descent", "mcts"),            # Cross: Descent vs MCTS
                    ("policy_only", "descent"),     # Cross: Policy-Only vs Descent
                    ("gumbel_mcts", "descent"),     # Cross: Gumbel vs Descent
                ]
            else:
                # 3p/4p: Include multiplayer-specific algorithms (MaxN, BRS, Paranoid MCTS)
                ai_type_combos = [
                    ("descent", "descent"),         # Both use descent (paranoid)
                    ("mcts", "mcts"),               # Both use MCTS (paranoid)
                    ("maxn", "maxn"),               # Both use MaxN (each maximizes own score)
                    ("brs", "brs"),                 # Both use BRS (best-reply search)
                    ("policy_only", "policy_only"), # Both use Policy-Only
                    ("gumbel_mcts", "gumbel_mcts"), # Both use Gumbel MCTS
                    ("mcts", "maxn"),               # Cross: MCTS vs MaxN
                    ("descent", "maxn"),            # Cross: Descent vs MaxN
                    ("brs", "descent"),             # Cross: BRS vs Descent
                    ("maxn", "brs"),                # Cross: MaxN vs BRS
                ]
            combo_idx = game_num % len(ai_type_combos)
            ai_type_a, ai_type_b = ai_type_combos[combo_idx]
        else:
            ai_type_a = ai_type_b = nn_ai_type

        # Run game with retry logic for transient failures
        result = None
        for attempt in range(game_retries):
            try:
                if is_baseline_match:
                    # Use generic model-vs-model for baseline players
                    result = play_model_vs_model_game(
                        model_a=play_a,
                        model_b=play_b,
                        board_type=board_type_enum,
                        num_players=num_players,
                        max_moves=10000,
                        save_game_history=jsonl_enabled,
                        game_seed=game_num,  # Jan 2026: unique randomness per game
                    )
                else:
                    # Use NN-specific game play for neural networks
                    result = play_nn_vs_nn_game(
                        model_a_path=play_a["model_path"],
                        model_b_path=play_b["model_path"],
                        board_type=board_type_enum,
                        num_players=num_players,
                        max_moves=10000,
                        mcts_simulations=50,  # Faster games
                        save_game_history=jsonl_enabled,
                        ai_type_a=ai_type_a,
                        ai_type_b=ai_type_b,
                    )
                break  # Success
            except FileNotFoundError as e:
                print(f"Skipping game: {e}")
                break  # Don't retry missing files
            except Exception as e:
                if attempt < game_retries - 1:
                    delay = exponential_backoff_delay(attempt, base_delay=0.5, max_delay=5.0)
                    print(f"Game {game_num} failed (attempt {attempt + 1}/{game_retries}): {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"Game {game_num} failed after {game_retries} attempts: {e}")
                    results["errors"] += 1

        if result is None:
            continue

        if result.get("error") or result.get("winner") == "error":
            results["errors"] += 1
            continue

        # Skip games where neural net fallback was detected
        if result.get("winner") == "skipped" or result.get("termination_reason") == "nn_fallback":
            print(f"  [SKIP] Game skipped due to NN fallback: {result.get('fallback_models', [])}")
            results["errors"] += 1  # Count as error for stats
            continue

        # Save game record to JSONL for training data
        game_record = result.get("game_record")
        if jsonl_path and game_record:
            game_record["tournament_id"] = tournament_id
            game_record["matchup_id"] = matchup_id
            game_record["game_num"] = game_num
            try:
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(game_record, cls=GameRecordEncoder) + "\n")
                    f.flush()
            except Exception as e:
                print(f"Warning: Failed to save game record: {e}")

        # Map back to original model_a/model_b
        winner_id = None
        if result["winner"] == "model_a":
            winner_id = id_a
        elif result["winner"] == "model_b":
            winner_id = id_b

        # Update stats based on original model_a vs model_b
        if winner_id == model_a["model_id"]:
            results["model_a_wins"] += 1
            winner = model_a["model_id"]
        elif winner_id == model_b["model_id"]:
            results["model_b_wins"] += 1
            winner = model_b["model_id"]
        else:
            results["draws"] += 1
            winner = "draw"

        if recording_config and result.get("recordable", True):
            game_ai_type_a = play_a.get("ai_type") or ai_type_a
            game_ai_type_b = play_b.get("ai_type") or ai_type_b
            record_metadata = dict(recording_base)
            record_metadata.update({
                "tournament_id": tournament_id,
                "matchup_id": matchup_id,
                "game_num": game_num,
                "model_a_id": model_a["model_id"],
                "model_b_id": model_b["model_id"],
                "seat_model_a_id": id_a,
                "seat_model_b_id": id_b,
                "model_a_path": play_a.get("model_path"),
                "model_b_path": play_b.get("model_path"),
                "ai_type_a": game_ai_type_a,
                "ai_type_b": game_ai_type_b,
                "ai_type_mode": "both" if use_both_ai_types else nn_ai_type,
                "baseline_match": is_baseline_match,
                "winner": winner,
                "winner_id": winner_id or "draw",
                "game_length": result.get("game_length", 0),
                "duration_sec": result.get("duration_sec", 0.0),
                "termination_reason": result.get("termination_reason"),
            })
            _record_game_if_enabled(
                recording_config=recording_config,
                initial_state=result.get("initial_state"),
                final_state=result.get("final_state"),
                moves=result.get("moves"),
                game_id=result.get("game_id"),
                extra_metadata=record_metadata,
                lock=recording_lock,
            )

        # Update Elo and record match using unified database
        # When use_composite is True, create composite participant IDs (nn:algo:config)
        participant_a = model_a["model_id"]
        participant_b = model_b["model_id"]

        if use_composite and HAS_COMPOSITE and make_composite_participant_id:
            # Get the AI type used for this specific game
            game_ai_type_a = play_a.get("ai_type") or ai_type_a
            game_ai_type_b = play_b.get("ai_type") or ai_type_b

            # Map seat AI types back to the fixed model ids
            seat_ai_types = {id_a: game_ai_type_a, id_b: game_ai_type_b}
            model_a_ai_type = seat_ai_types.get(model_a["model_id"], game_ai_type_a)
            model_b_ai_type = seat_ai_types.get(model_b["model_id"], game_ai_type_b)

            baseline_ai_types = {"random", "heuristic", "mcts", "gmo", "ebmo"}
            is_baseline_a = model_a.get("model_path", "").startswith("__BASELINE") or model_a.get("ai_type") in baseline_ai_types
            is_baseline_b = model_b.get("model_path", "").startswith("__BASELINE") or model_b.get("ai_type") in baseline_ai_types

            config_a = STANDARD_ALGORITHM_CONFIGS.get(model_a_ai_type, {})
            config_b = STANDARD_ALGORITHM_CONFIGS.get(model_b_ai_type, {})

            # Only create composite IDs for neural network models (not baselines)
            if not is_baseline_a:
                participant_a = make_composite_participant_id(
                    nn_id=model_a["model_id"],
                    ai_type=model_a_ai_type,
                    config=config_a,
                )
            if not is_baseline_b:
                participant_b = make_composite_participant_id(
                    nn_id=model_b["model_id"],
                    ai_type=model_b_ai_type,
                    config=config_b,
                )

            if not is_baseline_a:
                _register_composite_participant(
                    db=db,
                    participant_id=participant_a,
                    model=model_a,
                    ai_type=model_a_ai_type,
                    config=config_a,
                )
            if not is_baseline_b:
                _register_composite_participant(
                    db=db,
                    participant_id=participant_b,
                    model=model_b,
                    ai_type=model_b_ai_type,
                    config=config_b,
                )

            # Update winner to use composite ID if needed
            if winner == model_a["model_id"]:
                winner = participant_a
            elif winner == model_b["model_id"]:
                winner = participant_b

        update_elo_after_match(
            db,
            participant_a,
            participant_b,
            winner,
            board_type,
            num_players,
            tournament_id,
            game_length=result.get("game_length", 0),
            duration_sec=result.get("duration_sec", 0.0),
        )

    return results


# ============================================
# Persistent Elo Database
# ============================================

# Canonical Elo database - tracks trained model ratings and history
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
LEGACY_ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"  # Legacy, merged into unified

# Import unified Elo database module
import contextlib

from app.tournament import EloDatabase

UNIFIED_DB_AVAILABLE = True


def init_elo_database(db_path: Path = ELO_DB_PATH) -> EloDatabase:
    """Initialize unified Elo database.

    Uses the unified EloDatabase class for consistent schema across all tournament scripts.
    """
    return EloDatabase(db_path)


def discover_models(
    models_dir: Path,
    board_type: str = "square8",
    num_players: int = 2,
    include_nnue: bool = False,
    validate_compatibility: bool = True,
) -> list[dict[str, Any]]:
    """Discover all trained models for a given board type.

    Args:
        models_dir: Directory to search for NN models
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, 4)
        include_nnue: If True, also discover NNUE models in models/nnue/
        validate_compatibility: If True, validate each model's architecture matches board_type
    """
    models = []

    # Look for .pth files matching the board/player config
    # Generate patterns for various naming conventions
    base_pattern = board_type.replace("square", "sq").replace("hexagonal", "hex")
    pattern = f"{base_pattern}_{num_players}p"
    # Also match hex8 naming for hexagonal boards
    alt_patterns = []
    if board_type == "hexagonal":
        alt_patterns = [f"hex_{num_players}p", f"hex8_{num_players}p", f"hexagonal_{num_players}p"]

    for f in models_dir.glob("*.pth"):
        name = f.stem

        # Check if it matches the board/player pattern FIRST (for performance)
        # For ringrift_v models, validate board type compatibility
        if "ringrift_v" in name:
            # ringrift_v models need board type validation
            name_lower = name.lower()
            if board_type == "square8":
                # Accept sq8, square8, or no explicit board type (legacy default)
                if "sq19" in name_lower or "square19" in name_lower or "hex" in name_lower:
                    continue
            elif board_type == "square19":
                if "sq19" not in name_lower and "square19" not in name_lower:
                    continue
            elif board_type in ("hexagonal", "hex8"):
                if "hex" not in name_lower:
                    continue
        else:
            # Check main pattern and alternate patterns
            matches_pattern = pattern in name or any(p in name for p in alt_patterns)
            if not matches_pattern:
                continue

        # Skip NNUE-style checkpoints that use sparse feature architecture.
        # These have names like "policy_sq8_2p_*" and contain accumulator/hidden layers
        # that are incompatible with NeuralNetAI's spatial encoding.
        # NNUE checkpoints should be in models/nnue/ and use .pt extension.
        if name.startswith("policy_") and pattern in name:
            # This is likely an NNUE policy checkpoint, skip it
            # (proper NNUE support would require a different AI class)
            continue

        # Extract version info from filename or checkpoint metadata
        version = "unknown"
        if "ringrift_v5" in name or "_v5_" in name or name.endswith("_v5"):
            version = "v5"
        elif "ringrift_v4" in name or "_v4_" in name or name.endswith("_v4"):
            version = "v4"
        elif "ringrift_v3" in name or "_v3_" in name or name.endswith("_v3"):
            version = "v3"
        elif "nn_baseline" in name:
            version = "baseline"

        # Only read checkpoint metadata for specific models where we need to confirm version
        # Skip for performance - reading 2000+ checkpoints is too slow
        # The actual model architecture is detected when loaded for play

        # Add matching model (skip if file was deleted between glob and stat)
        try:
            stat_info = f.stat()
            models.append({
                "model_id": name,
                "model_path": str(f),
                "board_type": board_type,
                "num_players": num_players,
                "version": version,
                "size_mb": stat_info.st_size / (1024 * 1024),
                "created_at": stat_info.st_mtime,
                "model_type": "nn",
            })
        except (FileNotFoundError, OSError):
            # File was deleted or is a broken symlink - skip it
            continue

    # Also discover NNUE models if requested
    if include_nnue:
        nnue_dir = models_dir / "nnue"
        if nnue_dir.exists():
            nnue_pattern = f"nnue_{board_type}_{num_players}p"
            for f in nnue_dir.glob("*.pt"):
                name = f.stem
                # Check if it matches the board/player pattern
                if nnue_pattern in name or f"nnue_policy_{board_type}_{num_players}p" in name:
                    models.append({
                        "model_id": f"nnue_{name}",
                        "model_path": str(f),
                        "board_type": board_type,
                        "num_players": num_players,
                        "version": "nnue",
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "created_at": f.stat().st_mtime,
                        "model_type": "nnue",
                        "ai_type": "nnue",
                    })

    # Validate model compatibility if requested
    if validate_compatibility and models:
        from app.models import BoardType

        # Map string board_type to BoardType enum
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(board_type.lower())

        if board_type_enum:
            valid_models = []
            skipped_count = 0
            for model in models:
                is_valid, reason = validate_model_for_board(
                    model["model_path"], board_type_enum, num_players
                )
                if is_valid:
                    valid_models.append(model)
                else:
                    skipped_count += 1
                    # Only log first few to avoid spam
                    if skipped_count <= 5:
                        print(f"  [SKIP] {model['model_id']}: {reason}")
                    elif skipped_count == 6:
                        print(f"  [SKIP] ... and more (suppressing further messages)")

            if skipped_count > 0:
                print(f"  Filtered out {skipped_count} incompatible models")
            models = valid_models

    return sorted(models, key=lambda x: x["created_at"], reverse=True)


def get_baseline_players(board_type: str, num_players: int) -> list[dict[str, Any]]:
    """Get baseline player definitions for Elo calibration.

    These provide anchor points for the Elo scale:
    - random: PINNED at 400 ELO (anchor point for calibration)
    - heuristic: ~1200-1400 Elo (decent baseline)
    - mcts_100: ~1400-1600 Elo (strong baseline)
    """
    now = time.time()
    baselines = [
        {
            "model_id": f"baseline_random_{board_type}_{num_players}p",
            "model_path": "__BASELINE_RANDOM__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "random",
        },
        {
            "model_id": f"baseline_heuristic_{board_type}_{num_players}p",
            "model_path": "__BASELINE_HEURISTIC__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "heuristic",
        },
        {
            "model_id": f"baseline_mcts_100_{board_type}_{num_players}p",
            "model_path": "__BASELINE_MCTS_100__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "mcts",
            "mcts_simulations": 100,
        },
        {
            "model_id": f"baseline_mcts_500_{board_type}_{num_players}p",
            "model_path": "__BASELINE_MCTS_500__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "mcts",
            "mcts_simulations": 500,
        },
        {
            "model_id": f"baseline_gmo_{board_type}_{num_players}p",
            "model_path": "__BASELINE_GMO__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "gmo",
        },
        {
            "model_id": f"baseline_ebmo_{board_type}_{num_players}p",
            "model_path": "__BASELINE_EBMO__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "ebmo",
        },
    ]
    return baselines


def register_models(db: EloDatabase, models: list[dict[str, Any]]):
    """Register discovered models in the database.

    Optimized for large model counts:
    - Fetches existing participants in a single query
    - Skips already-registered models
    - Uses batch insert for new models
    - Emits heartbeat progress to prevent SSH timeout
    """
    if not models:
        return

    # Step 1: Fetch existing participant IDs in a single query (fast)
    print(f"[register] Checking {len(models)} models against database...")
    sys.stdout.flush()

    conn = db._get_connection()
    model_ids = [m["model_id"] for m in models]

    # Batch query for existing IDs using IN clause (chunked for SQLite limit)
    existing_ids = set()
    chunk_size = 500  # SQLite SQLITE_MAX_VARIABLE_NUMBER is typically 999
    for i in range(0, len(model_ids), chunk_size):
        chunk = model_ids[i:i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT participant_id FROM participants WHERE participant_id IN ({placeholders})",
            chunk
        ).fetchall()
        existing_ids.update(row[0] for row in rows)

    # Step 2: Filter to only new models
    new_models = [m for m in models if m["model_id"] not in existing_ids]

    if not new_models:
        print(f"[register] All {len(models)} models already registered")
        sys.stdout.flush()
        return

    print(f"[register] {len(existing_ids)} already registered, {len(new_models)} new models to add")
    sys.stdout.flush()

    # Step 3: Prepare batch insert data
    now = time.time()
    insert_data = []
    for idx, m in enumerate(new_models):
        # Emit heartbeat every 100 models to prevent SSH timeout
        if idx > 0 and idx % 100 == 0:
            print(f"[register] Preparing model {idx}/{len(new_models)}...")
            sys.stdout.flush()

        model_path = m.get("model_path", "")
        if model_path.startswith("__BASELINE_RANDOM__"):
            ai_type = "random"
            participant_type = "baseline"
        elif model_path.startswith("__BASELINE_HEURISTIC__"):
            ai_type = "heuristic"
            participant_type = "baseline"
        elif model_path.startswith("__BASELINE_MCTS"):
            ai_type = "mcts"
            participant_type = "baseline"
        elif model_path.startswith("__BASELINE_GMO__"):
            ai_type = "gmo"
            participant_type = "baseline"
        elif model_path.startswith("__BASELINE_EBMO__"):
            ai_type = "ebmo"
            participant_type = "baseline"
        else:
            ai_type = m.get("ai_type", "neural_net")
            participant_type = "model"

        insert_data.append((
            m["model_id"],
            participant_type,
            ai_type,
            None,  # difficulty
            ai_type == "neural_net",  # use_neural_net
            m.get("model_path"),
            m.get("version"),
            None,  # metadata
            now,  # created_at
            now,  # last_seen
        ))

    # Step 4: Bulk insert using executemany (much faster than individual inserts)
    print(f"[register] Bulk inserting {len(insert_data)} new models...")
    sys.stdout.flush()

    conn.executemany("""
        INSERT INTO participants
        (participant_id, participant_type, ai_type, difficulty, use_neural_net,
         model_path, model_version, metadata, created_at, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(participant_id) DO UPDATE SET
            last_seen = excluded.last_seen
    """, insert_data)
    conn.commit()

    print(f"[register] Done: registered {len(new_models)} new models in bulk")
    sys.stdout.flush()


def get_leaderboard(
    db: EloDatabase,
    board_type: str | None = None,
    num_players: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get current Elo leaderboard."""
    rows = db.get_leaderboard(board_type=board_type, num_players=num_players, min_games=0, limit=limit)

    results = []
    for row in rows:
        games = row.get("games_played", 0)
        wins = row.get("wins", 0)
        win_rate = (wins / games * 100) if games > 0 else 0

        results.append({
            "rank": len(results) + 1,
            "model_id": row.get("participant_id"),  # Keep as model_id for backward compat
            "participant_id": row.get("participant_id"),
            "board_type": row.get("board_type"),
            "num_players": row.get("num_players"),
            "rating": round(row.get("rating", 1500.0), 1),
            "games_played": games,
            "wins": wins,
            "losses": row.get("losses", 0),
            "draws": row.get("draws", 0),
            "win_rate": round(win_rate, 1),
            "version": row.get("model_version", "unknown"),
            "last_update": (datetime.fromtimestamp(float(row["last_update"])).isoformat()
                           if row.get("last_update") and str(row["last_update"]).replace('.','').isdigit()
                           else row.get("last_update")),
        })

    return results


def update_elo_after_match(
    db: EloDatabase,
    model_a: str,
    model_b: str,
    winner: str,  # model_a, model_b, or "draw"
    board_type: str,
    num_players: int,
    tournament_id: str | None = None,
    k_factor: float = 32.0,
    game_length: int = 0,
    duration_sec: float = 0.0,
):
    """Update Elo ratings after a match using unified EloDatabase."""
    # Get old Elo ratings before update (for event emission)
    old_rating_a = db.get_rating(model_a, board_type, num_players)
    old_rating_b = db.get_rating(model_b, board_type, num_players)
    old_elo_a = old_rating_a.rating
    old_elo_b = old_rating_b.rating

    # Determine rankings from winner
    if winner == model_a:
        is_draw = False
    elif winner == model_b:
        # Swap so winner is first in the call
        model_a, model_b = model_b, model_a
        old_elo_a, old_elo_b = old_elo_b, old_elo_a
        is_draw = False
    else:
        is_draw = True

    db.record_two_player_result(
        winner_id=model_a,
        loser_id=model_b,
        board_type=board_type,
        num_players=num_players,
        tournament_id=tournament_id or "model_elo_tournament",
        is_draw=is_draw,
        game_length=game_length,
        duration_sec=duration_sec,
    )

    # Emit ELO_UPDATED events for feedback loop integration
    if HAS_EVENT_BUS:
        config_key = f"{board_type}_{num_players}p"
        rating_a = db.get_rating(model_a, board_type, num_players)
        rating_b = db.get_rating(model_b, board_type, num_players)
        new_elo_a = rating_a.rating
        new_elo_b = rating_b.rating
        games_a = rating_a.games_played
        games_b = rating_b.games_played

        # Run async event emission in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, schedule the coroutines
                asyncio.ensure_future(emit_elo_updated(
                    config=config_key, model_id=model_a, new_elo=new_elo_a,
                    old_elo=old_elo_a, games_played=games_a, source=GAME_SOURCE_TAG
                ))
                asyncio.ensure_future(emit_elo_updated(
                    config=config_key, model_id=model_b, new_elo=new_elo_b,
                    old_elo=old_elo_b, games_played=games_b, source=GAME_SOURCE_TAG
                ))
            else:
                # No running loop, run synchronously
                loop.run_until_complete(emit_elo_updated(
                    config=config_key, model_id=model_a, new_elo=new_elo_a,
                    old_elo=old_elo_a, games_played=games_a, source=GAME_SOURCE_TAG
                ))
                loop.run_until_complete(emit_elo_updated(
                    config=config_key, model_id=model_b, new_elo=new_elo_b,
                    old_elo=old_elo_b, games_played=games_b, source=GAME_SOURCE_TAG
                ))
        except (RuntimeError, asyncio.TimeoutError, asyncio.CancelledError, TypeError, AttributeError):
            # Don't fail the match update if event emission fails
            pass


def print_leaderboard(leaderboard: list[dict[str, Any]], title: str = "Elo Leaderboard"):
    """Pretty print the leaderboard."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

    if not leaderboard:
        print("  No models found in leaderboard.")
        return

    print(f"{'Rank':<6}{'Model':<50}{'Elo':>8}{'Games':>8}{'Win%':>8}{'Type':>10}")
    print("-" * 90)

    for entry in leaderboard:
        # Support both model_id and participant_id keys
        model_id = entry.get("model_id") or entry.get("participant_id", "unknown")
        model_short = model_id[:48] if len(model_id) > 48 else model_id
        version = entry.get("version") or entry.get("participant_type", "?")
        print(f"{entry['rank']:<6}{model_short:<50}{entry['rating']:>8.1f}{entry['games_played']:>8}{entry['win_rate']:>7.1f}%{version:>10}")

    print(f"\nTotal models: {len(leaderboard)}")


# All supported board/player configurations for Elo tracking
ALL_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]


def run_all_config_tournaments(
    args,
    *,
    recording_metadata_base: dict[str, Any],
    jsonl_enabled: bool,
    jsonl_dir: Path | None,
    recording_lock: threading.Lock | None,
):
    """Run tournaments for all board/player configurations.

    This ensures there's an Elo ranking for each combination of board type and number of players.
    """
    import uuid

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    db = init_elo_database(db_path)
    models_dir = AI_SERVICE_ROOT / "models"
    record_db_enabled = not args.no_record_db and is_recording_enabled()

    print(f"\n{'='*80}")
    print(" Running Elo Tournaments for All Configurations")
    print(f"{'='*80}")

    overall_start = time.time()
    total_games_all = 0

    for board_type, num_players in ALL_CONFIGS:
        config_label = f"{board_type} {num_players}p"
        print(f"\n{'='*60}")
        print(f" Configuration: {config_label}")
        print(f"{'='*60}")

        # Discover models for this config
        if args.baselines_only:
            models = get_baseline_players(board_type, num_players)
            print(f"Using {len(models)} baseline players for {config_label}")
        else:
            models = discover_models(models_dir, board_type, num_players)
            print(f"Discovered {len(models)} models for {config_label}")
            if args.include_baselines and not args.no_baselines:
                baselines = get_baseline_players(board_type, num_players)
                models.extend(baselines)
                print(f"Added {len(baselines)} baseline players (required for ELO anchoring)")

        if args.top_n and not args.baselines_only:
            models = models[:args.top_n]
            print(f"Using top {args.top_n} most recent models")

        if len(models) < 2:
            print(f"  Skipping {config_label}: need at least 2 models")
            continue

        # Register models
        register_models(db, models)

        if args.leaderboard_only:
            leaderboard = get_leaderboard(db, board_type, num_players)
            print_leaderboard(leaderboard, f"Elo Leaderboard - {config_label}")
            continue

        if not args.run:
            # Just show plan
            matchups = []
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    matchups.append((m1, m2))
            print(f"  Would run {len(matchups) * args.games} games ({len(matchups)} matchups × {args.games} games)")
            continue

        # Run tournament for this config
        tournament_id = str(uuid.uuid4())[:8]
        config_recording = None
        if record_db_enabled:
            config_recording = _build_recording_config(
                board_type,
                num_players,
                source_tag=GAME_SOURCE_TAG,
                db_dir=args.record_db_dir,
                db_prefix=args.record_db_prefix,
                db_path=None,
                tags=_recording_tags_for_args(args, board_type, num_players),
            )
        config_metadata = dict(recording_metadata_base)
        config_metadata["config"] = f"{board_type}_{num_players}p"
        matchups = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                matchups.append((m1, m2))

        print(f"Running tournament {tournament_id}: {len(matchups)} matchups × {args.games} games")

        config_start = time.time()
        games_completed = 0

        for matchup_idx, (m1, m2) in enumerate(matchups):
            print(f"  [{matchup_idx + 1}/{len(matchups)}] {m1['model_id'][:30]} vs {m2['model_id'][:30]}", end=" ")

            try:
                results = run_model_matchup(
                    db=db,
                    model_a=m1,
                    model_b=m2,
                    board_type=board_type,
                    num_players=num_players,
                    games=args.games,
                    tournament_id=tournament_id,
                    nn_ai_type=args.ai_type,
                    use_both_ai_types=args.both_ai_types,
                    save_games_dir=jsonl_dir,
                    jsonl_enabled=jsonl_enabled,
                    recording_config=config_recording,
                    recording_metadata_base=config_metadata,
                    recording_lock=recording_lock,
                    use_composite=getattr(args, "composite", False),
                    game_retries=args.game_retries,
                )
                games_completed += args.games
                print(f"A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

        config_elapsed = time.time() - config_start
        total_games_all += games_completed
        print(f"  Completed {games_completed} games in {config_elapsed:.1f}s")

        # Show updated leaderboard
        leaderboard = get_leaderboard(db, board_type, num_players, limit=10)
        print_leaderboard(leaderboard, f"Top 10 - {config_label}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(" All Tournaments Complete")
    print(f"{'='*80}")
    print(f"Total games: {total_games_all}")
    print(f"Total time: {overall_elapsed:.1f}s")

    db.close()


def run_continuous_tournament(
    args,
    *,
    recording_config: RecordingConfig | None,
    recording_metadata_base: dict[str, Any],
    jsonl_enabled: bool,
    jsonl_dir: Path | None,
    recording_lock: threading.Lock | None,
):
    """Run tournaments continuously in daemon mode.

    This provides continuous model evaluation for the unified AI improvement loop:
    - Shadow tournaments every 15 minutes (10 games)
    - Full tournaments every hour (50 games)
    - Checkpoint monitoring for new models
    - Event emission for pipeline integration
    """
    import signal

    # Try to import event bus for pipeline integration
    emit_events = args.emit_events
    if emit_events:
        try:
            import asyncio

            from app.coordination.event_router import (
                emit_error,
                emit_evaluation_completed,
            )
        except ImportError:
            print("[ContinuousTournament] Warning: Event bus not available, disabling event emission")
            emit_events = False

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    db = init_elo_database(db_path)
    models_dir = AI_SERVICE_ROOT / "models"

    interval = args.continuous_interval
    checkpoint_watch_dir = Path(args.checkpoint_watch) if args.checkpoint_watch else None
    known_checkpoints = set()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[ContinuousTournament] Shutdown requested")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[ContinuousTournament] Starting continuous evaluation daemon")
    print(f"  Interval: {interval}s ({interval // 60} min)")
    print(f"  Games per matchup: {args.games}")
    print(f"  Board: {args.board}, Players: {args.players}")
    if checkpoint_watch_dir:
        print(f"  Watching: {checkpoint_watch_dir}")

    iteration = 0
    while running:
        iteration += 1
        iteration_start = time.time()

        print(f"\n{'='*60}")
        print(f"[ContinuousTournament] Iteration {iteration} at {datetime.now().isoformat()}")
        print(f"{'='*60}")

        try:
            # Check for new checkpoints if watching
            if checkpoint_watch_dir and checkpoint_watch_dir.exists():
                for pth_file in checkpoint_watch_dir.glob("*.pth"):
                    file_key = f"{pth_file}:{pth_file.stat().st_mtime}"
                    if file_key not in known_checkpoints:
                        known_checkpoints.add(file_key)
                        print(f"[ContinuousTournament] New checkpoint: {pth_file.name}")

            # Discover models
            if args.baselines_only:
                models = get_baseline_players(args.board, args.players)
            else:
                models = discover_models(models_dir, args.board, args.players, include_nnue=args.include_nnue)
                if args.include_baselines and not args.no_baselines:
                    models.extend(get_baseline_players(args.board, args.players))

            if args.top_n:
                models = models[:args.top_n]

            if len(models) < 2:
                print(f"[ContinuousTournament] Not enough models ({len(models)}), waiting...")
                time.sleep(interval)
                continue

            # Register models
            register_models(db, models)

            # Filter archived models
            active_models = [m for m in models if not is_model_archived(db, m["model_id"], args.board, args.players)]
            if len(active_models) < 2:
                print(f"[ContinuousTournament] Not enough active models ({len(active_models)}), waiting...")
                time.sleep(interval)
                continue
            models = active_models

            # Generate matchups
            if args.elo_matchmaking:
                matchups = generate_elo_based_matchups(models, db, args.board, args.players, args.elo_range)
            else:
                matchups = []
                for i, m1 in enumerate(models):
                    for m2 in models[i+1:]:
                        matchups.append((m1, m2))

            # Limit matchups for shadow mode
            if args.quick and len(matchups) > 10:
                # Sample matchups for quick evaluation
                import random
                matchups = random.sample(matchups, min(10, len(matchups)))

            print(f"[ContinuousTournament] Running {len(matchups)} matchups × {args.games} games")

            import uuid
            tournament_id = f"cont_{str(uuid.uuid4())[:8]}"
            iteration_metadata = dict(recording_metadata_base)
            iteration_metadata["mode"] = "continuous"
            iteration_metadata["iteration"] = iteration
            games_completed = 0
            total_wins = 0
            total_games = 0

            for matchup_idx, (m1, m2) in enumerate(matchups):
                if not running:
                    break

                try:
                    results = run_model_matchup(
                        db=db,
                        model_a=m1,
                        model_b=m2,
                        board_type=args.board,
                        num_players=args.players,
                        games=args.games,
                        tournament_id=tournament_id,
                        nn_ai_type=args.ai_type,
                        use_both_ai_types=args.both_ai_types,
                        save_games_dir=jsonl_dir,
                        jsonl_enabled=jsonl_enabled,
                        recording_config=recording_config,
                        recording_metadata_base=iteration_metadata,
                        recording_lock=recording_lock,
                        use_composite=getattr(args, "composite", False),
                        game_retries=args.game_retries,
                    )
                    games_completed += args.games
                    total_wins += results["model_a_wins"] + results["model_b_wins"]
                    total_games += results["model_a_wins"] + results["model_b_wins"] + results["draws"]
                    print(f"  [{matchup_idx + 1}/{len(matchups)}] A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']}")

                except Exception as e:
                    print(f"  [{matchup_idx + 1}/{len(matchups)}] ERROR: {e}")
                    continue

            iteration_duration = time.time() - iteration_start

            # Get best model Elo for reporting
            leaderboard = get_leaderboard(db, args.board, args.players, limit=1)
            best_elo = leaderboard[0]["rating"] if leaderboard else 1500.0
            win_rate = total_wins / total_games if total_games > 0 else 0.5

            print(f"[ContinuousTournament] Completed {games_completed} games in {iteration_duration:.1f}s")
            print(f"  Best Elo: {best_elo:.0f}, Win rate: {win_rate:.1%}")

            # Emit event for pipeline integration
            if emit_events:
                config_key = f"{args.board}_{args.players}p"
                try:
                    asyncio.run(emit_evaluation_completed(
                        config=config_key,
                        elo=best_elo,
                        games_played=games_completed,
                        win_rate=win_rate,
                        source=GAME_SOURCE_TAG,
                    ))
                except Exception as e:
                    print(f"[ContinuousTournament] Failed to emit event: {e}")

            # Print summary output for parsing by shadow tournament service
            print(f"Win rate: {win_rate * 100:.1f}%")
            print(f"Elo: {best_elo:.0f}")

        except Exception as e:
            print(f"[ContinuousTournament] Error in iteration: {e}")
            import traceback
            traceback.print_exc()

            if emit_events:
                with contextlib.suppress(Exception):
                    asyncio.run(emit_error(
                        component="continuous_tournament",
                        error=str(e),
                        source=GAME_SOURCE_TAG,
                    ))

        # Wait for next iteration
        if running:
            elapsed = time.time() - iteration_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                print(f"[ContinuousTournament] Sleeping {sleep_time:.0f}s until next iteration")
                try:
                    time.sleep(sleep_time)
                except KeyboardInterrupt:
                    running = False

    print("[ContinuousTournament] Stopped")
    db.close()


# Archive and matchmaking functions moved to scripts/lib/tournament_cli.py
# Imports: archive_low_elo_models, filter_archived_models, generate_elo_based_matchups,
#          is_model_archived, unarchive_discovered_models, unarchive_model


def main():
    parser = argparse.ArgumentParser(description="Run model Elo tournament")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--games", type=int, default=10, help="Games per matchup")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers for matchup execution (default: CPU//2)")
    parser.add_argument("--single-threaded", action="store_true", help="Force single-threaded mode for debugging")
    parser.add_argument("--top-n", type=int, help="Only include top N models by recency")
    parser.add_argument("--top-elo", type=int, help="Only include top N models by ELO rating (recommended for focused evaluation)")
    parser.add_argument("--composite", action="store_true", help="Use composite participant IDs (nn:algo:config) for per-algorithm ELO tracking")
    parser.add_argument("--leaderboard-only", action="store_true", help="Just show leaderboard")
    parser.add_argument("--run", action="store_true", help="Actually run games (otherwise just shows plan)")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--db", type=str, help="DEPRECATED: Always uses unified_elo.db for data integrity. This flag is ignored.")
    parser.add_argument("--all-configs", action="store_true", help="Run tournament for all board/player configurations")
    parser.add_argument("--elo-matchmaking", action="store_true", help="Use Elo-based matchmaking (pair similar-rated models)")
    parser.add_argument("--elo-range", type=int, default=200, help="Max Elo difference for matchmaking (default: 200)")
    parser.add_argument("--archive-threshold", type=int, default=ARCHIVE_ELO_THRESHOLD, help=f"Archive models below this Elo after 50+ games (default: {ARCHIVE_ELO_THRESHOLD})")
    parser.add_argument("--archive", action="store_true", help="Archive low-Elo models")
    parser.add_argument("--include-baselines", action="store_true", default=True,
                        help="Include baseline players (Random, Heuristic, MCTS) - DEFAULT: ON for ELO calibration")
    parser.add_argument("--no-baselines", action="store_true", help="Exclude baseline players (not recommended - breaks ELO anchoring)")
    parser.add_argument("--baselines-only", action="store_true", help="Run tournament with only baseline players (for calibration)")
    parser.add_argument("--ai-type", choices=[
        "random", "heuristic", "minimax", "gpu_minimax", "mcts", "descent",
        "policy_only", "gumbel_mcts", "maxn", "brs", "neural_demo",
        "gmo", "gmo_mcts", "ebmo", "ig_gmo"
    ], default="descent", help="AI type for neural networks when --no-both-ai-types is used (default: descent). Includes experimental AIs: GMO, GMO-MCTS, EBMO, IG-GMO for 2000 Elo target.")
    parser.add_argument("--both-ai-types", action="store_true", default=True, help="Use ALL AI types (MCTS, Descent, Policy-Only, Gumbel MCTS) for comprehensive NN evaluation (DEFAULT: ON)")
    parser.add_argument("--no-both-ai-types", action="store_true", help="Disable multi-AI-type evaluation, use only --ai-type")
    parser.add_argument("--include-nnue", action="store_true", help="Include NNUE models from models/nnue/ directory")
    parser.add_argument("--nnue-only", action="store_true", help="Run tournament with only NNUE models")

    # Continuous evaluation modes (for unified AI loop integration)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 games per matchup for fast shadow evaluation")
    parser.add_argument("--continuous", action="store_true", help="Continuous daemon mode: run tournaments periodically")
    parser.add_argument("--continuous-interval", type=int, default=900, help="Interval between continuous runs (seconds, default: 900)")
    parser.add_argument("--checkpoint-watch", type=str, help="Directory to watch for new model checkpoints")
    parser.add_argument("--emit-events", action="store_true", help="Emit data events for pipeline integration")

    # Training data generation mode
    parser.add_argument("--training-mode", action="store_true",
                        help="Training mode: tag games as 'elo_selfplay' so they feed into training pool instead of being filtered as holdout")

    # Recording / export controls
    parser.add_argument("--no-record-db", action="store_true",
                        help="Disable GameReplayDB recording (canonical replay)")
    parser.add_argument("--record-db-dir", type=str, default="data/games",
                        help="Directory for tournament GameReplayDB files")
    parser.add_argument("--record-db-prefix", type=str, default="tournament",
                        help="Database filename prefix for tournament recordings")
    parser.add_argument("--record-db-path", type=str,
                        help="Explicit GameReplayDB path (single-config only)")
    parser.add_argument("--no-jsonl", action="store_true",
                        help="Disable legacy JSONL output")
    parser.add_argument("--jsonl-dir", type=str,
                        help="Directory for JSONL exports (defaults to data/holdouts/elo_tournaments)")
    parser.add_argument("--shard-id", type=str, help="Optional shard identifier for metadata")
    parser.add_argument("--worker-id", type=str, help="Optional worker identifier for metadata")
    parser.add_argument("--export-npz", action="store_true",
                        help="Export training NPZ from the recorded tournament DB after the run")
    parser.add_argument("--npz-output", type=str,
                        help="Output path for tournament NPZ (default: data/training/elo_tournament_<board>_<players>p.npz)")

    # Performance optimization
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile() for faster startup (reduces per-game overhead at cost of inference speed)")

    # Resilience options
    parser.add_argument("--game-retries", type=int, default=3,
                        help="Number of retries for transient game failures (default: 3)")
    parser.add_argument("--matchup-timeout", type=int, default=DEFAULT_MATCHUP_TIMEOUT,
                        help=f"Timeout per matchup in seconds (default: {DEFAULT_MATCHUP_TIMEOUT})")
    parser.add_argument("--tournament-timeout", type=int, default=DEFAULT_TOURNAMENT_TIMEOUT,
                        help=f"Timeout for entire tournament in seconds (default: {DEFAULT_TOURNAMENT_TIMEOUT})")

    args = parser.parse_args()

    # Determine worker count (parallelism is the default)
    if args.single_threaded:
        args.workers = 1
    elif args.workers is None:
        args.workers = get_tournament_workers()

    # Handle --no-both-ai-types flag (overrides default True for --both-ai-types)
    if args.no_both_ai_types:
        args.both_ai_types = False

    # Handle --no-compile flag to disable torch.compile() for faster startup
    if args.no_compile:
        os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"
        print("[Tournament] torch.compile() disabled for faster startup")

    # === TRAINING MODE: Change source tag so games feed into training pool ===
    global GAME_SOURCE_TAG
    if args.training_mode:
        GAME_SOURCE_TAG = "elo_selfplay"
        print("[Tournament] Training mode enabled: games will be tagged as 'elo_selfplay' for training pool inclusion")

    import socket
    node_id = socket.gethostname()
    worker_id = str(args.worker_id or os.getpid())
    jsonl_enabled = not args.no_jsonl
    jsonl_dir = Path(args.jsonl_dir) if args.jsonl_dir else None

    record_db_path = args.record_db_path
    if args.all_configs and record_db_path:
        print("[Tournament] --record-db-path ignored with --all-configs; using prefix/dir per config")
        record_db_path = None

    record_db_enabled = not args.no_record_db and is_recording_enabled()
    if args.no_record_db:
        print("[Tournament] GameReplayDB recording disabled via --no-record-db")
    elif not record_db_enabled:
        print("[Tournament] GameReplayDB recording disabled via RINGRIFT_RECORD_SELFPLAY_GAMES")

    recording_metadata_base = {
        "node_id": node_id,
        "worker_id": worker_id,
        "runner": "run_model_elo_tournament",
    }
    if args.shard_id:
        recording_metadata_base["shard_id"] = args.shard_id

    recording_config = None
    if record_db_enabled:
        recording_config = _build_recording_config(
            args.board,
            args.players,
            source_tag=GAME_SOURCE_TAG,
            db_dir=args.record_db_dir,
            db_prefix=args.record_db_prefix,
            db_path=record_db_path,
            tags=_recording_tags_for_args(args, args.board, args.players),
        )

    # === PROCESS SAFEGUARDS ===
    # Limit torch.compile workers to prevent process sprawl
    cpu_count = os.cpu_count() or 8
    # Use conservative worker limits based on CPU count
    max_torch_workers = max(2, min(cpu_count // 4, 4))
    max_omp_threads = max(1, min(cpu_count // 8, 2))
    os.environ.setdefault("TORCH_COMPILE_MAX_WORKERS", str(max_torch_workers))
    os.environ.setdefault("OMP_NUM_THREADS", str(max_omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_omp_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_omp_threads))

    # Check system load before starting - be conservative
    try:
        load_avg = os.getloadavg()[0]
        load_threshold_warn = max(cpu_count * 0.5, 20)  # Warn at 50% of CPU count
        load_threshold_error = max(cpu_count * 0.8, 40)  # Error at 80% of CPU count

        if load_avg > load_threshold_warn and not args.leaderboard_only:
            print(f"[Tournament] WARNING: System load is {load_avg:.1f} (threshold: {load_threshold_warn:.0f})")
            print("[Tournament] Consider waiting for load to decrease or use --leaderboard-only")
            if load_avg > load_threshold_error:
                print(f"[Tournament] ERROR: Load too high (>{load_threshold_error:.0f}), aborting to prevent system overload")
                print("[Tournament] Tip: Kill other heavy processes or wait for them to finish")
                return
    except (OSError, AttributeError):
        pass  # getloadavg not available on all platforms

    # Check coordination for exclusive tournament access
    task_id = None
    coord_start_time = time.time()
    if HAS_COORDINATION and not args.leaderboard_only:
        # Check if another tournament is running
        try:
            registry = get_registry()
            if registry.is_role_held(OrchestratorRole.TOURNAMENT_RUNNER):
                holder = registry.get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER)
                holder_pid = holder.pid if holder else "unknown"
                print(f"[Tournament] ERROR: Another tournament is already running (PID {holder_pid})")
                print("[Tournament] Wait for it to complete or kill it first")
                return
        except Exception as e:
            print(f"[Tournament] Registry check warning: {e}")

        # Duration-aware scheduling: check if tournament can be scheduled now
        can_schedule, schedule_reason = can_schedule_task("tournament", node_id)
        if not can_schedule:
            print(f"[Tournament] Deferred by duration scheduler: {schedule_reason}")
            print("[Tournament] Use --leaderboard-only to view current standings")
            return

        # Estimate tournament duration
        est_duration = estimate_task_duration("tournament", config=f"{args.board}_{args.players}p")
        eta_time = datetime.fromtimestamp(time.time() + est_duration).strftime("%H:%M:%S")
        print(f"[Tournament] Estimated duration: {est_duration/60:.0f} min (ETA: {eta_time})")

        # Register task for tracking
        task_id = f"tournament_{args.board}_{args.players}p_{os.getpid()}"
        try:
            register_running_task(task_id, "tournament", node_id, os.getpid())
            print(f"[Tournament] Registered task {task_id}")
        except Exception as e:
            print(f"[Tournament] Warning: Failed to register task: {e}")
    # === END SAFEGUARDS ===

    # Quick mode: reduce games for fast shadow evaluation
    if args.quick:
        args.games = 10
        args.run = True  # Quick mode implies --run

    # Continuous mode: run as daemon
    if args.continuous:
        run_continuous_tournament(
            args,
            recording_config=recording_config,
            recording_metadata_base=recording_metadata_base,
            jsonl_enabled=jsonl_enabled,
            jsonl_dir=jsonl_dir,
            recording_lock=_RECORDING_LOCK,
        )
        return

    # If --all-configs, loop through all configurations
    if args.all_configs:
        run_all_config_tournaments(
            args,
            recording_metadata_base=recording_metadata_base,
            jsonl_enabled=jsonl_enabled,
            jsonl_dir=jsonl_dir,
            recording_lock=_RECORDING_LOCK,
        )
        return

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    print(f"[main] Initializing Elo database at {db_path}...")
    sys.stdout.flush()
    db = init_elo_database(db_path)

    # Discover models
    print("[main] Scanning for models...")
    sys.stdout.flush()
    models_dir = AI_SERVICE_ROOT / "models"
    if args.baselines_only:
        models = get_baseline_players(args.board, args.players)
        print(f"\nUsing {len(models)} baseline players for {args.board} {args.players}p")
    elif args.nnue_only:
        # Only NNUE models
        models = discover_models(models_dir, args.board, args.players, include_nnue=True)
        models = [m for m in models if m.get("model_type") == "nnue"]
        print(f"\nDiscovered {len(models)} NNUE models for {args.board} {args.players}p")
    else:
        models = discover_models(models_dir, args.board, args.players, include_nnue=args.include_nnue)
        nn_count = len([m for m in models if m.get("model_type") != "nnue"])
        nnue_count = len([m for m in models if m.get("model_type") == "nnue"])
        print(f"\nDiscovered {nn_count} NN models for {args.board} {args.players}p")
        if nnue_count > 0:
            print(f"Also found {nnue_count} NNUE models")
        if args.include_baselines and not args.no_baselines:
            baselines = get_baseline_players(args.board, args.players)
            models.extend(baselines)
            print(f"Added {len(baselines)} baseline players (required for ELO anchoring)")
        elif args.no_baselines:
            print("WARNING: Baselines excluded - ELO ratings may drift without anchor!")

    if args.top_n and not args.baselines_only:
        models = models[:args.top_n]
        print(f"Using top {args.top_n} most recent models")

    # --top-elo: Select top N models by ELO rating (more useful for focused evaluation)
    if args.top_elo and not args.baselines_only:
        # Get current ELO ratings for all models using batch query
        print(f"[main] Fetching Elo ratings for {len(models)} models...")
        sys.stdout.flush()
        model_ids = [m.get("model_id", m.get("participant_id", "")) for m in models]
        ratings_map = db.get_ratings_batch(model_ids, args.board, args.players)
        model_elos = []
        for m in models:
            mid = m.get("model_id", m.get("participant_id", ""))
            rating = ratings_map.get(mid)
            model_elos.append((m, rating.rating if rating else 1500.0))
        # Sort by ELO descending and take top N
        model_elos.sort(key=lambda x: x[1], reverse=True)
        # Separate baselines (keep all) from NN models (filter to top N)
        baseline_ai_types = {"random", "heuristic", "mcts", "gmo", "ebmo"}
        baselines = [m for m, _ in model_elos if m.get("ai_type") in baseline_ai_types]
        nn_models = [(m, elo) for m, elo in model_elos if m.get("ai_type") not in baseline_ai_types]
        top_nn = [m for m, _ in nn_models[:args.top_elo]]
        models = top_nn + baselines
        print(f"\n--top-elo {args.top_elo} selection breakdown:")
        print(f"  Total models before selection: {len(model_elos)}")
        print(f"  Baselines (all kept): {len(baselines)}")
        print(f"  NN models found: {len(nn_models)}")
        print(f"  Top NN selected: {len(top_nn)}")
        if top_nn:
            top_idx = min(args.top_elo - 1, len(nn_models) - 1)
            print(f"  Top NN ELO range: {nn_models[0][1]:.0f} - {nn_models[top_idx][1]:.0f}")
            for i, (m, elo) in enumerate(nn_models[:min(5, args.top_elo)]):
                print(f"    #{i+1}: {m['model_id'][:50]} (ELO: {elo:.0f})")
        print(f"Using top {args.top_elo} models by ELO rating + {len(baselines)} baselines")

    # Register models
    register_models(db, models)

    # Auto-unarchive discovered models (handles case where files restored from archived/)
    unarchived = unarchive_discovered_models(db, models, args.board, args.players)
    if unarchived > 0:
        print(f"Auto-unarchived {unarchived} models (files exist in models/)")

    # Filter out archived models (only those not present in filesystem)
    active_models = [m for m in models if not is_model_archived(db, m["model_id"], args.board, args.players)]
    if len(active_models) < len(models):
        print(f"Filtered out {len(models) - len(active_models)} archived models")
        models = active_models

    # Handle archiving if requested
    if args.archive:
        archived = archive_low_elo_models(
            db, args.board, args.players,
            elo_threshold=args.archive_threshold,
            min_games=50,
        )
        if archived:
            print(f"\nArchived {len(archived)} low-Elo models:")
            for model_id in archived:
                print(f"  - {model_id}")
            # Re-filter models
            models = [m for m in models if m["model_id"] not in archived]

    # Show leaderboard
    leaderboard = get_leaderboard(db, args.board, args.players)
    print_leaderboard(leaderboard, f"Current Elo Leaderboard - {args.board} {args.players}p")

    if args.leaderboard_only:
        db.close()
        return

    if len(models) < 2:
        print("\nNeed at least 2 models to run a tournament!")
        db.close()
        return

    # Generate matchups (Elo-based or round-robin)
    if args.elo_matchmaking:
        print(f"\nUsing Elo-based matchmaking (max diff: {args.elo_range})")
        matchups = generate_elo_based_matchups(
            models, db, args.board, args.players, args.elo_range
        )
    else:
        # Standard round-robin matchups
        matchups = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                matchups.append((m1, m2))

    print(f"\n{'='*80}")
    print(" Tournament Plan")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Matchups: {len(matchups)}")
    print(f"Games per matchup: {args.games}")
    print(f"Total games needed: {len(matchups) * args.games}")

    # Check if --run flag provided
    if not args.run:
        print("\nSample matchups:")
        for m1, m2 in matchups[:5]:
            print(f"  {m1['model_id'][:40]} vs {m2['model_id'][:40]}")

        if len(matchups) > 5:
            print(f"  ... and {len(matchups) - 5} more")

        print("\nAdd --run flag to execute games and update Elo ratings.")
        db.close()
        return

    # Enable tournament mode with cache sized for all models
    # This prevents cache thrashing when playing many games with a fixed set of models
    from app.ai.model_cache import set_tournament_mode
    set_tournament_mode(True, max_models=len(models) + 10)

    # Run the tournament
    import uuid
    tournament_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*80}")
    print(f" Running Tournament {tournament_id}")
    print(f"{'='*80}")
    if args.workers > 1:
        print(f"Parallel execution with {args.workers} workers")

    total_games = len(matchups) * args.games
    games_completed = 0
    start_time = time.time()

    # Helper function for parallel execution
    def run_single_matchup(matchup_data):
        matchup_idx, m1, m2 = matchup_data
        try:
            results = run_model_matchup(
                db=db,  # EloDatabase uses thread-local connections
                model_a=m1,
                model_b=m2,
                board_type=args.board,
                num_players=args.players,
                games=args.games,
                tournament_id=tournament_id,
                nn_ai_type=args.ai_type,
                use_both_ai_types=args.both_ai_types,
                save_games_dir=jsonl_dir,
                jsonl_enabled=jsonl_enabled,
                recording_config=recording_config,
                recording_metadata_base=recording_metadata_base,
                recording_lock=_RECORDING_LOCK,
                use_composite=getattr(args, "composite", False),
                game_retries=args.game_retries,
            )
            return (matchup_idx, m1, m2, results, None)
        except Exception as e:
            import traceback
            return (matchup_idx, m1, m2, None, str(e))

    if args.workers > 1:
        # Parallel execution with ThreadPoolExecutor
        matchup_data = [(i, m1, m2) for i, (m1, m2) in enumerate(matchups)]

        # Heartbeat tracking for stuck detection
        last_completion_time = time.time()
        matchups_completed = 0
        matchups_timed_out = 0
        heartbeat_stop = threading.Event()

        def heartbeat_thread():
            """Log progress periodically to detect stuck tournaments."""
            while not heartbeat_stop.is_set():
                heartbeat_stop.wait(HEARTBEAT_INTERVAL)
                if heartbeat_stop.is_set():
                    break
                elapsed = time.time() - start_time
                since_last = time.time() - last_completion_time
                remaining_timeout = args.tournament_timeout - elapsed
                print(
                    f"[Heartbeat] Tournament running for {elapsed:.0f}s, "
                    f"{matchups_completed}/{len(matchups)} matchups done, "
                    f"{since_last:.0f}s since last completion, "
                    f"timeout in {remaining_timeout:.0f}s"
                )
                if since_last > args.matchup_timeout:
                    print(f"[Warning] No matchup completed in {since_last:.0f}s (timeout: {args.matchup_timeout}s)")

        heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
        heartbeat.start()

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_single_matchup, data): data for data in matchup_data}
                pending_futures = set(futures.keys())

                # Use tournament timeout for overall as_completed
                tournament_deadline = start_time + args.tournament_timeout

                while pending_futures:
                    # Calculate remaining time for this batch
                    remaining_time = tournament_deadline - time.time()
                    if remaining_time <= 0:
                        print(f"\n[Timeout] Tournament timeout ({args.tournament_timeout}s) exceeded")
                        # Cancel remaining futures
                        for f in pending_futures:
                            f.cancel()
                        break

                    # Use matchup timeout for individual completion waits
                    try:
                        done_iter = as_completed(pending_futures, timeout=min(args.matchup_timeout, remaining_time))
                        for future in done_iter:
                            pending_futures.discard(future)
                            last_completion_time = time.time()

                            try:
                                matchup_idx, m1, m2, results, error = future.result(timeout=5)
                            except Exception as e:
                                print(f"\n[Error] Failed to get matchup result: {e}")
                                continue

                            matchups_completed += 1

                            if error:
                                print(f"\nMatchup {matchup_idx + 1}/{len(matchups)}: {m1['model_id'][:35]} vs {m2['model_id'][:35]}")
                                print(f"  Error: {error}")
                                continue

                            games_completed += args.games
                            elapsed = time.time() - start_time
                            rate = games_completed / elapsed if elapsed > 0 else 0

                            print(f"\nMatchup {matchup_idx + 1}/{len(matchups)}: {m1['model_id'][:35]} vs {m2['model_id'][:35]}")
                            print(f"  Results: A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']} E={results['errors']}")
                            print(f"  Progress: {games_completed}/{total_games} games ({rate:.1f} games/sec)")

                    except FuturesTimeoutError:
                        matchups_timed_out += 1
                        print(f"\n[Timeout] Matchup batch timed out after {args.matchup_timeout}s ({matchups_timed_out} timeouts so far)")
                        # Cancel oldest pending futures if we hit too many timeouts
                        if matchups_timed_out >= 3:
                            print(f"[Warning] {matchups_timed_out} timeouts - cancelling stuck futures")
                            for f in list(pending_futures)[:args.workers]:
                                f.cancel()
                                pending_futures.discard(f)

                if pending_futures:
                    print(f"\n[Warning] {len(pending_futures)} matchups did not complete (cancelled/timed out)")
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
    else:
        # Sequential execution (original behavior)
        for matchup_idx, (m1, m2) in enumerate(matchups):
            print(f"\nMatchup {matchup_idx + 1}/{len(matchups)}: {m1['model_id'][:35]} vs {m2['model_id'][:35]}")

            try:
                results = run_model_matchup(
                    db=db,
                    model_a=m1,
                    model_b=m2,
                    board_type=args.board,
                    num_players=args.players,
                    games=args.games,
                    tournament_id=tournament_id,
                    nn_ai_type=args.ai_type,
                    use_both_ai_types=args.both_ai_types,
                    save_games_dir=jsonl_dir,
                    jsonl_enabled=jsonl_enabled,
                    recording_config=recording_config,
                    recording_metadata_base=recording_metadata_base,
                    recording_lock=_RECORDING_LOCK,
                    use_composite=getattr(args, "composite", False),
                    game_retries=args.game_retries,
                )

                games_completed += args.games
                elapsed = time.time() - start_time
                rate = games_completed / elapsed if elapsed > 0 else 0

                print(f"  Results: A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']} E={results['errors']}")
                print(f"  Progress: {games_completed}/{total_games} games ({rate:.1f} games/sec)")

            except Exception as e:
                import traceback
                print(f"  Error in matchup: {e}")
                traceback.print_exc()
                continue

    # Show final leaderboard
    final_leaderboard = get_leaderboard(db, args.board, args.players, limit=100)
    print_leaderboard(final_leaderboard, f"Final Elo Leaderboard - {args.board} {args.players}p (Tournament {tournament_id})")

    # Summary
    elapsed = time.time() - start_time
    print(f"\nTournament completed in {elapsed:.1f} seconds")
    print(f"Total games played: {games_completed}")

    db.close()

    if args.export_npz:
        if args.all_configs or args.continuous:
            print("[Tournament] --export-npz is only supported for single-run tournaments")
        elif recording_config is None:
            print("[Tournament] No GameReplayDB recording available; cannot export NPZ")
        else:
            try:
                from scripts.export_replay_dataset import export_replay_dataset

                if args.board == "square19":
                    board_type_enum = BoardType.SQUARE19
                elif args.board in ("hex", "hexagonal"):
                    board_type_enum = BoardType.HEXAGONAL
                elif args.board == "hex8":
                    board_type_enum = BoardType.HEX8
                else:
                    board_type_enum = BoardType.SQUARE8

                output_path = args.npz_output
                if not output_path:
                    output_path = str(
                        AI_SERVICE_ROOT
                        / "data"
                        / "training"
                        / f"elo_tournament_{args.board}_{args.players}p.npz"
                    )
                export_replay_dataset(
                    db_path=recording_config.get_db_path(),
                    board_type=board_type_enum,
                    num_players=args.players,
                    output_path=output_path,
                    require_completed=True,
                )
            except Exception as e:
                print(f"[Tournament] NPZ export failed: {e}")

    # Restore normal cache mode and clean up
    from app.ai.model_cache import clear_model_cache as clear_cache
    set_tournament_mode(False)
    clear_cache()

    # Record task completion for duration learning
    if HAS_COORDINATION and task_id:
        try:
            config = f"{args.board}_{args.players}p"
            # Args: task_type, host, started_at, completed_at, success, config
            record_task_completion("tournament", node_id, coord_start_time, time.time(), True, config)
            print("[Tournament] Recorded task completion")
        except Exception as e:
            print(f"[Tournament] Warning: Failed to record task completion: {e}")


if __name__ == "__main__":
    main()
