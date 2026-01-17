"""Unified Game Gauntlet Module for RingRift AI Evaluation.

Consolidates game-playing logic that was duplicated across:
- scripts/select_best_checkpoint_by_elo.py
- app/training/tier_eval_runner.py
- app/training/background_eval.py

This module provides:
- play_single_game(): Play one game between two AIs
- run_baseline_gauntlet(): Evaluate a model against baseline opponents
- BaselineOpponent: Enum of standard baseline types

Usage:
    from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

    results = run_baseline_gauntlet(
        model_path="models/my_model.pth",
        board_type=BoardType.SQUARE8,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        games_per_opponent=20,
    )
    print(f"Win rate vs random: {results['random']['win_rate']:.1%}")
"""

from __future__ import annotations

import logging
import random
import sqlite3
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.training.env import get_theoretical_max_moves
from app.training.composite_participant import make_composite_participant_id
from app.utils.parallel_defaults import get_parallel_games_default

logger = logging.getLogger(__name__)


# ============================================
# Default Gauntlet Recording Configuration (Dec 2025)
# ============================================

def _create_gauntlet_recording_config(
    board_type: Any,
    num_players: int,
    source: str = "gauntlet",
) -> Any:
    """Create a RecordingConfig for gauntlet games.

    Returns None if recording is disabled or not available.
    Games are saved to data/games/gauntlet_{board_type}_{num_players}p.db
    """
    try:
        from app.db.unified_recording import RecordingConfig, is_recording_enabled

        if not is_recording_enabled():
            return None

        board_value = getattr(board_type, "value", str(board_type))
        db_path = f"data/games/gauntlet_{board_value}_{num_players}p.db"

        return RecordingConfig(
            board_type=board_type,
            num_players=num_players,
            db_path=db_path,
            source=source,
            enabled=True,
        )
    except ImportError:
        return None
    except (AttributeError, TypeError, ValueError, OSError) as e:
        # AttributeError: board_type has no .value attribute
        # TypeError: RecordingConfig constructor type mismatch
        # ValueError: Config validation failed
        # OSError: Database path issues
        logger.debug(f"[gauntlet] Could not create recording config: {e}")
        return None

# Import model cache for cleanup after gauntlet runs (Dec 29, 2025)
try:
    from app.ai.model_cache import clear_model_cache
    HAS_MODEL_CACHE = True
except ImportError:
    HAS_MODEL_CACHE = False
    clear_model_cache = None  # type: ignore


# ============================================
# NNUE Model Auto-Discovery (Jan 13, 2026)
# ============================================

def _get_nnue_model_path(board_type: Any, num_players: int) -> str | None:
    """Auto-discover NNUE model for the given config.

    This function explicitly finds NNUE model paths to ensure NNUE baselines
    actually load NNUE models instead of falling back to heuristic evaluation.

    Args:
        board_type: Board type (enum or string)
        num_players: Number of players

    Returns:
        Path to NNUE model if found, None otherwise
    """
    import os

    # Get board type value string
    if hasattr(board_type, "value"):
        board_str = board_type.value
    else:
        board_str = str(board_type).lower()

    # Search paths for NNUE models
    base_paths = [
        "models/nnue",
        "models",
        "data/models/nnue",
    ]

    # Try various naming conventions
    name_patterns = [
        f"nnue_{board_str}_{num_players}p.pt",
        f"nnue_{board_str}_{num_players}p.pth",
        f"canonical_nnue_{board_str}_{num_players}p.pt",
    ]

    for base_path in base_paths:
        for pattern in name_patterns:
            full_path = os.path.join(base_path, pattern)
            if os.path.exists(full_path):
                logger.debug(f"[gauntlet] Found NNUE model at {full_path}")
                return full_path

    logger.debug(f"[gauntlet] No NNUE model found for {board_str}_{num_players}p")
    return None

# Lazy imports to avoid circular dependencies and heavy imports at module load
# Note: TYPE_CHECKING imports removed - using Any for lazy-loaded modules
_torch_loaded = False
_game_modules_loaded = False

# Declare globals for lazy-loaded modules (actual imports happen in _ensure_game_modules)
BoardType: Any = None
AIType: Any = None
AIConfig: Any = None
GameStatus: Any = None
HeuristicAI: Any = None
RandomAI: Any = None
PolicyOnlyAI: Any = None
UniversalAI: Any = None
create_initial_state: Any = None
DefaultRulesEngine: Any = None


def _ensure_game_modules():
    """Lazy load game-related modules."""
    global _game_modules_loaded
    if _game_modules_loaded:
        return

    global BoardType, AIType, AIConfig, GameStatus
    global HeuristicAI, RandomAI, PolicyOnlyAI, UniversalAI
    global create_initial_state, DefaultRulesEngine

    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.policy_only_ai import PolicyOnlyAI
    from app.ai.random_ai import RandomAI
    from app.ai.universal_ai import UniversalAI
    from app.models import AIConfig, AIType, BoardType, GameStatus
    from app.rules.default_engine import DefaultRulesEngine
    from app.training.initial_state import create_initial_state

    _game_modules_loaded = True


# ============================================
# Model Architecture Verification (Dec 30, 2025)
# ============================================
# Catches model mismatches before wasting gauntlet time


def verify_model_architecture(
    model_path: str | Path,
    board_type: Any,
    num_players: int,
) -> tuple[bool, str]:
    """Verify model architecture matches expected config.

    Checks that the value head output dimension matches the expected player count.
    This prevents models trained for N players from being loaded for M players,
    which would cause partial weight loading and degraded performance.

    Args:
        model_path: Path to the model checkpoint
        board_type: Board type (for logging context)
        num_players: Expected number of players (determines value head size)

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.

    Example:
        >>> is_valid, error = verify_model_architecture("models/hex8_4p.pth", "hex8", 4)
        >>> if not is_valid:
        ...     logger.error(f"Architecture mismatch: {error}")
    """
    try:
        from app.utils.torch_utils import safe_load_checkpoint
    except ImportError:
        # Fallback if safe_load_checkpoint not available
        import torch
        safe_load_checkpoint = lambda p: torch.load(p, map_location="cpu", weights_only=False)

    path = Path(model_path)
    if not path.exists():
        return False, f"Model file not found: {path}"

    try:
        checkpoint = safe_load_checkpoint(path)
    except Exception as e:
        return False, f"Failed to load checkpoint: {e}"

    # Get the model state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        return False, "Checkpoint does not contain model_state_dict"

    # Check value head dimensions - find the FINAL value layer
    # v2 models have: value_fc1, value_fc2 (output)
    # v5-heavy models have: value_fc1, value_fc2, value_fc3 (output)
    import re
    value_fc_pattern = re.compile(r"value_fc(\d+)\.weight$")
    value_layers = {}
    for key in state_dict.keys():
        match = value_fc_pattern.search(key)
        if match:
            layer_num = int(match.group(1))
            value_layers[layer_num] = key

    if not value_layers:
        # Model doesn't have standard value head naming - skip check
        logger.debug(f"[verify_arch] No value_fc*.weight found in {path.name}, skipping check")
        return True, ""

    # Get the highest-numbered layer (the final output layer)
    final_layer_num = max(value_layers.keys())
    final_layer_key = value_layers[final_layer_num]
    final_layer_weight = state_dict[final_layer_key]
    actual_outputs = final_layer_weight.shape[0]
    expected_outputs = num_players

    if actual_outputs != expected_outputs:
        board_str = getattr(board_type, "value", str(board_type)) if board_type else "unknown"
        return False, (
            f"Model architecture mismatch: value_fc{final_layer_num} has {actual_outputs} outputs, "
            f"expected {expected_outputs} for {num_players}-player {board_str} config. "
            f"Model may have been trained for {actual_outputs}-player games."
        )

    return True, ""


# ============================================
# Adaptive Resource Management (Dec 2025)
# ============================================
# Prevents resource exhaustion by scaling workers based on system load


def get_adaptive_max_workers(requested: int = 4) -> int:
    """Reduce workers if system is under load.

    This helps prevent resource exhaustion when multiple gauntlets run
    concurrently or when the system is already under heavy load.

    Args:
        requested: Requested number of workers (default: 4)

    Returns:
        Adjusted worker count based on system load
    """
    try:
        import os
        load_avg = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        load_ratio = load_avg / cpu_count

        if load_ratio > 0.8:
            # Heavy load: use minimal workers
            return max(1, requested // 4)
        elif load_ratio > 0.5:
            # Moderate load: use half workers
            return max(2, requested // 2)
        else:
            # Normal load: use requested workers
            return requested
    except (OSError, AttributeError):
        # If we can't check load, use conservative default
        return min(requested, 2)


def _allow_parallel_opponents() -> bool:
    """Return True if it's safe to use parallel opponent evaluation."""
    try:
        import multiprocessing as mp
        main_module = sys.modules.get("__main__")
        main_file = getattr(main_module, "__file__", None)
        if not main_file:
            return False
        return mp.current_process().name == "MainProcess"
    except (OSError, AttributeError, RuntimeError) as e:
        return False


class BaselineOpponent(Enum):
    """Standard baseline opponents for evaluation.

    December 2025: Extended with stronger tiers to measure Elo above 1600.
    Previous baselines (RANDOM@400, HEURISTIC@1200) capped measurable Elo.

    Baseline Elo ladder:
        RANDOM:            ~400 Elo (random legal moves)
        HEURISTIC:        ~1200 Elo (handcrafted evaluation, difficulty 5)
        HEURISTIC_STRONG: ~1400 Elo (difficulty 8, deeper search)
        MCTS_LIGHT:       ~1500 Elo (MCTS with 32 simulations)
        NNUE_BRS_D3:      ~1550 Elo (NNUE + BRS depth 3, multiplayer)
        NNUE_MINIMAX_D4:  ~1600 Elo (NNUE + Minimax depth 4, 2-player)
        NNUE_MAXN_D3:     ~1650 Elo (NNUE + MaxN depth 3, multiplayer)
        MCTS_MEDIUM:      ~1700 Elo (MCTS with 128 simulations)
        MCTS_STRONG:      ~1900 Elo (MCTS with 512 simulations)
        MCTS_MASTER:      ~2000 Elo (MCTS with 1024 simulations) - Dec 28
        MCTS_GRANDMASTER: ~2100 Elo (MCTS with 2048 simulations) - Dec 28

    December 2025: Added NNUE baselines for unified harness evaluation.
    NNUE baselines evaluate models under different search harnesses:
        - NNUE_MINIMAX_D4: Best for 2-player games (alpha-beta pruning)
        - NNUE_MAXN_D3: Accurate multiplayer (each player maximizes)
        - NNUE_BRS_D3: Fast multiplayer (greedy best reply)
    """
    RANDOM = "random"
    HEURISTIC = "heuristic"
    HEURISTIC_STRONG = "heuristic_strong"
    MCTS_LIGHT = "mcts_light"
    MCTS_MEDIUM = "mcts_medium"
    MCTS_STRONG = "mcts_strong"
    MCTS_MASTER = "mcts_master"              # Dec 28: 2000+ Elo
    MCTS_GRANDMASTER = "mcts_grandmaster"    # Dec 28: 2100+ Elo
    # Dec 29: NNUE baseline opponents for unified harness evaluation
    NNUE_MINIMAX_D4 = "nnue_minimax_d4"      # NNUE under minimax depth 4
    NNUE_MAXN_D3 = "nnue_maxn_d3"            # NNUE under MaxN depth 3 (3+ players)
    NNUE_BRS_D3 = "nnue_brs_d3"              # NNUE under BRS depth 3 (3+ players)

    # Dec 29: Phase 6 - Extended harness types for comprehensive evaluation
    # PolicyOnly variants (~1100-1300 Elo depending on model quality)
    POLICY_ONLY_NN = "policy_only_nn"        # Full NN policy head, greedy selection
    POLICY_ONLY_NNUE = "policy_only_nnue"    # NNUE policy head, greedy selection

    # Gumbel MCTS variants (~1400-1800 Elo, higher budget = stronger)
    GUMBEL_B64 = "gumbel_b64"                # Gumbel MCTS budget 64 (~1400 Elo)
    GUMBEL_B200 = "gumbel_b200"              # Gumbel MCTS budget 200 (~1600 Elo)
    GUMBEL_NNUE = "gumbel_nnue"              # Gumbel with NNUE evaluation (~1500 Elo)

    # Descent variants (~1300-1500 Elo, gradient-based search)
    DESCENT_NN = "descent_nn"                # Descent with Full NN evaluation
    DESCENT_NNUE = "descent_nnue"            # Descent with NNUE evaluation

    # GPU-accelerated variants (Jan 2026) - high throughput evaluation
    GPU_GUMBEL = "gpu_gumbel"                # GPU Gumbel MCTS (~1600 Elo, 2-10x faster)


# ============================================
# Early Stopping with Statistical Confidence (Dec 2025)
# ============================================


def compute_wilson_interval(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for win rate.

    Wilson score interval is more accurate than normal approximation,
    especially for extreme proportions or small samples.

    Args:
        wins: Number of wins
        total: Total games played
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound) for win rate
    """
    if total == 0:
        return (0.0, 1.0)

    try:
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
    except ImportError:
        # Fallback z-scores for common confidence levels
        z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_table.get(confidence, 1.96)

    p_hat = wins / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * ((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) ** 0.5 / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def should_early_stop(
    wins: int,
    losses: int,
    threshold: float = 0.5,
    confidence: float = 0.95,
    min_games: int = 10,
) -> tuple[bool, str]:
    """Check if we can stop early based on statistical confidence.

    Uses Wilson score interval to determine if the win rate is significantly
    above or below the threshold with the given confidence level.

    Args:
        wins: Number of wins so far
        losses: Number of losses so far
        threshold: Win rate threshold to compare against (e.g., 0.5 for 50%)
        confidence: Confidence level (e.g., 0.95 for 95%)
        min_games: Minimum games before early stopping is considered

    Returns:
        Tuple of (should_stop, reason) where reason explains why/why not
    """
    total = wins + losses

    if total < min_games:
        return (False, f"Need {min_games - total} more games")

    lower, upper = compute_wilson_interval(wins, total, confidence)

    # Check if entire confidence interval is above threshold (clear win)
    if lower > threshold:
        return (True, f"Win rate {wins/total:.1%} (CI: {lower:.1%}-{upper:.1%}) > {threshold:.0%}")

    # Check if entire confidence interval is below threshold (clear loss)
    if upper < threshold:
        return (True, f"Win rate {wins/total:.1%} (CI: {lower:.1%}-{upper:.1%}) < {threshold:.0%}")

    return (False, f"Inconclusive: CI {lower:.1%}-{upper:.1%} spans {threshold:.0%}")


# Import baseline Elo estimates from centralized config
try:
    from app.config.thresholds import (
        BASELINE_ELO_HEURISTIC,
        BASELINE_ELO_RANDOM,
        # Dec 28, 2025: Added 2000+ Elo baselines
        BASELINE_ELO_MCTS_MASTER,
        BASELINE_ELO_MCTS_GRANDMASTER,
        GAUNTLET_GAMES_PER_OPPONENT,
        MIN_WIN_RATE_VS_HEURISTIC,
        MIN_WIN_RATE_VS_RANDOM,
        get_min_win_rate_vs_heuristic,
        get_min_win_rate_vs_random,
        get_elo_adaptive_win_rate_vs_heuristic,
        get_elo_adaptive_win_rate_vs_random,
        # Dec 30, 2025: Graduated thresholds based on game count
        get_game_count_based_win_rate_vs_random,
        get_game_count_based_win_rate_vs_heuristic,
        get_graduated_thresholds,
        get_game_count_tier,
    )
    HAS_ELO_ADAPTIVE = True
    HAS_GRADUATED_THRESHOLDS = True
    # Dec 29, 2025: NNUE baseline Elo ratings (not in thresholds.py yet)
    BASELINE_ELO_NNUE_MINIMAX_D4 = 1300
    BASELINE_ELO_NNUE_MAXN_D3 = 1250
    BASELINE_ELO_NNUE_BRS_D3 = 1250
except ImportError:
    # Fallback values - keep in sync with app/config/thresholds.py
    BASELINE_ELO_RANDOM = 400
    BASELINE_ELO_HEURISTIC = 1200
    # Dec 28, 2025: Added 2000+ Elo baselines
    BASELINE_ELO_MCTS_MASTER = 2000
    BASELINE_ELO_MCTS_GRANDMASTER = 2100
    # Dec 29, 2025: NNUE baseline Elo ratings
    BASELINE_ELO_NNUE_MINIMAX_D4 = 1300  # NNUE minimax depth 4
    BASELINE_ELO_NNUE_MAXN_D3 = 1250  # NNUE MaxN depth 3 (3+ players)
    BASELINE_ELO_NNUE_BRS_D3 = 1250  # NNUE BRS depth 3 (3+ players)
    GAUNTLET_GAMES_PER_OPPONENT = 50
    # Dec 29, 2025: Updated to match thresholds.py (rationalized for player count)
    # Using ~1.7x multiplier over random baseline consistently
    MIN_WIN_RATE_VS_RANDOM = 0.85  # 85% for 2p (1.7x over 50% baseline)
    MIN_WIN_RATE_VS_HEURISTIC = 0.85  # 85% for 2p
    MIN_WIN_RATE_VS_RANDOM_3P = 0.55  # 55% for 3p (1.65x over 33% baseline)
    MIN_WIN_RATE_VS_HEURISTIC_3P = 0.45  # 45% for 3p
    MIN_WIN_RATE_VS_RANDOM_4P = 0.45  # 45% for 4p (1.8x over 25% baseline)
    MIN_WIN_RATE_VS_HEURISTIC_4P = 0.35  # 35% for 4p
    HAS_ELO_ADAPTIVE = False

    def get_min_win_rate_vs_random(num_players: int = 2) -> float:
        """Get minimum win rate vs random based on player count."""
        if num_players >= 4:
            return MIN_WIN_RATE_VS_RANDOM_4P
        if num_players == 3:
            return MIN_WIN_RATE_VS_RANDOM_3P
        return MIN_WIN_RATE_VS_RANDOM

    def get_min_win_rate_vs_heuristic(num_players: int = 2) -> float:
        """Get minimum win rate vs heuristic based on player count."""
        if num_players >= 4:
            return MIN_WIN_RATE_VS_HEURISTIC_4P
        if num_players == 3:
            return MIN_WIN_RATE_VS_HEURISTIC_3P
        return MIN_WIN_RATE_VS_HEURISTIC

    def get_elo_adaptive_win_rate_vs_random(model_elo: float, num_players: int = 2) -> float:
        return get_min_win_rate_vs_random(num_players)

    HAS_GRADUATED_THRESHOLDS = False

    def get_game_count_tier(game_count: int) -> str:
        """Fallback tier calculation."""
        if game_count < 5000:
            return "bootstrap"
        elif game_count < 20000:
            return "standard"
        return "aspirational"

    def get_game_count_based_win_rate_vs_random(game_count: int, num_players: int = 2) -> float:
        """Fallback: use standard thresholds."""
        return get_min_win_rate_vs_random(num_players)

    def get_game_count_based_win_rate_vs_heuristic(game_count: int, num_players: int = 2) -> float:
        """Fallback: use standard thresholds."""
        return get_min_win_rate_vs_heuristic(num_players)

    def get_graduated_thresholds(game_count: int, num_players: int = 2) -> dict:
        """Fallback: use standard thresholds."""
        return {
            "tier": get_game_count_tier(game_count),
            "random": get_min_win_rate_vs_random(num_players),
            "heuristic": get_min_win_rate_vs_heuristic(num_players),
        }

    def get_elo_adaptive_win_rate_vs_heuristic(model_elo: float, num_players: int = 2) -> float:
        return get_min_win_rate_vs_heuristic(num_players)

# December 2025: Extended baseline Elo ladder
BASELINE_ELO_HEURISTIC_STRONG = 1400
BASELINE_ELO_MCTS_LIGHT = 1500
BASELINE_ELO_MCTS_MEDIUM = 1700
BASELINE_ELO_MCTS_STRONG = 1900

BASELINE_ELOS = {
    BaselineOpponent.RANDOM: BASELINE_ELO_RANDOM,
    BaselineOpponent.HEURISTIC: BASELINE_ELO_HEURISTIC,
    BaselineOpponent.HEURISTIC_STRONG: BASELINE_ELO_HEURISTIC_STRONG,
    BaselineOpponent.MCTS_LIGHT: BASELINE_ELO_MCTS_LIGHT,
    BaselineOpponent.MCTS_MEDIUM: BASELINE_ELO_MCTS_MEDIUM,
    BaselineOpponent.MCTS_STRONG: BASELINE_ELO_MCTS_STRONG,
    # Dec 28, 2025: Added 2000+ Elo baselines
    BaselineOpponent.MCTS_MASTER: BASELINE_ELO_MCTS_MASTER,
    BaselineOpponent.MCTS_GRANDMASTER: BASELINE_ELO_MCTS_GRANDMASTER,
    # Dec 29, 2025: NNUE baseline Elo ratings for unified harness evaluation
    BaselineOpponent.NNUE_MINIMAX_D4: BASELINE_ELO_NNUE_MINIMAX_D4,
    BaselineOpponent.NNUE_MAXN_D3: BASELINE_ELO_NNUE_MAXN_D3,
    BaselineOpponent.NNUE_BRS_D3: BASELINE_ELO_NNUE_BRS_D3,
}

# Static fallback (use get_min_win_rate_* functions for player-aware thresholds)
# For stronger baselines, 50% is the promotion threshold (beat at parity)
MIN_WIN_RATES = {
    BaselineOpponent.RANDOM: MIN_WIN_RATE_VS_RANDOM,
    BaselineOpponent.HEURISTIC: MIN_WIN_RATE_VS_HEURISTIC,
    BaselineOpponent.HEURISTIC_STRONG: 0.50,  # Beat at parity
    BaselineOpponent.MCTS_LIGHT: 0.50,
    BaselineOpponent.MCTS_MEDIUM: 0.50,
    BaselineOpponent.MCTS_STRONG: 0.50,
    # Dec 28, 2025: 2000+ Elo baselines - 50% to pass (beat at parity)
    BaselineOpponent.MCTS_MASTER: 0.50,
    BaselineOpponent.MCTS_GRANDMASTER: 0.50,
}


@dataclass
class GauntletGameResult:
    """Result of a single gauntlet game.

    December 2025: Renamed from GameResult to avoid collision with
    app.training.selfplay_runner.GameResult (canonical for selfplay) and
    app.execution.game_executor.GameResult (canonical for execution).
    """
    winner: int | None  # Player number who won, or None for draw
    move_count: int
    victory_reason: str
    candidate_player: int  # Which player was the candidate
    candidate_won: bool


# Backward-compat alias (deprecated Dec 2025)
GameResult = GauntletGameResult


@dataclass
class GauntletResult:
    """Aggregated results from a gauntlet evaluation."""
    total_games: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    win_rate: float = 0.0

    # Per-opponent results
    opponent_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Baseline gating
    passes_baseline_gating: bool = True
    failed_baselines: list[str] = field(default_factory=list)

    # Per-baseline threshold tracking (January 2026)
    thresholds_met: dict[str, bool] = field(default_factory=dict)

    # Elo estimate
    estimated_elo: float = 1500.0

    # Early stopping (December 2025)
    early_stopped_baselines: list[str] = field(default_factory=list)
    games_saved_by_early_stopping: int = 0

    # Harness metadata (December 2025 - Phase 2)
    harness_type: str = ""  # e.g., "gumbel_mcts", "minimax", "maxn"
    harness_config_hash: str = ""  # Configuration hash for Elo tracking
    model_id: str = ""  # Composite participant ID for Elo
    visit_distributions: list[dict[str, float]] = field(default_factory=list)  # For soft targets

    # Architecture verification (December 30, 2025)
    passed: bool = True  # True if gauntlet ran to completion
    failure_reason: str = ""  # Non-empty if passed=False (e.g., architecture mismatch)


class _HarnessToAIWrapper:
    """Wrapper that adapts AIHarness interface to the AI interface expected by play_single_game.

    Jan 2026: Created to allow harness-based opponents (GPU_GUMBEL, etc.) to work with
    the gauntlet's play_single_game function which expects AI instances with select_move().
    """

    def __init__(self, harness: Any, player_number: int) -> None:
        """Initialize the wrapper.

        Args:
            harness: An AIHarness instance (from app.ai.harness)
            player_number: Player number this AI represents
        """
        self.harness = harness
        self.player_number = player_number
        self._last_visit_distribution: tuple[list, list] | None = None

    def select_move(self, game_state: Any) -> Any:
        """Select a move using the harness.

        Args:
            game_state: Current game state

        Returns:
            The selected move
        """
        move, metadata = self.harness.evaluate(game_state, self.player_number)

        # Store visit distribution for extraction by play_single_game
        visit_dist = self.harness.get_visit_distribution()
        if visit_dist:
            moves = list(visit_dist.keys())
            probs = list(visit_dist.values())
            self._last_visit_distribution = (moves, probs)

        return move

    def get_visit_distribution(self) -> tuple[list, list] | None:
        """Get the visit distribution from the last search.

        Returns:
            Tuple of (moves, probabilities) or None if not available.
        """
        return self._last_visit_distribution

    def reset(self) -> None:
        """Reset internal state between games."""
        self._last_visit_distribution = None
        if hasattr(self.harness, 'reset'):
            self.harness.reset()


def create_baseline_ai(
    baseline: BaselineOpponent,
    player: int,
    board_type: Any,  # BoardType
    difficulty: int | None = None,
    game_seed: int | None = None,
    num_players: int = 2,
    model_path: str | Path | None = None,  # Jan 2026: For harness-based baselines
) -> Any:
    """Create an AI instance for a baseline opponent.

    Args:
        baseline: Which baseline to create
        player: Player number (1 or 2)
        board_type: Board type enum
        difficulty: Optional difficulty override
        game_seed: Optional seed for RNG variation per game
        num_players: Number of players in the game (for NNUE multiplayer baselines)
        model_path: Path to model checkpoint (for GPU_GUMBEL and other harness-based baselines)

    Returns:
        AI instance ready to play
    """
    _ensure_game_modules()

    # Derive player-specific seed for varied but reproducible behavior
    ai_rng_seed = None
    if game_seed is not None:
        ai_rng_seed = (game_seed * 104729 + player * 7919) & 0xFFFFFFFF

    if baseline == BaselineOpponent.RANDOM:
        config = AIConfig(
            ai_type=AIType.RANDOM,
            board_type=board_type,
            difficulty=difficulty or 1,
            rngSeed=ai_rng_seed,
        )
        return RandomAI(player, config)

    elif baseline == BaselineOpponent.HEURISTIC:
        config = AIConfig(
            ai_type=AIType.HEURISTIC,
            board_type=board_type,
            difficulty=difficulty or 5,
            rngSeed=ai_rng_seed,
        )
        return HeuristicAI(player, config)

    elif baseline == BaselineOpponent.HEURISTIC_STRONG:
        # December 2025: Strong heuristic (~1400 Elo)
        config = AIConfig(
            ai_type=AIType.HEURISTIC,
            board_type=board_type,
            difficulty=difficulty or 8,  # Higher difficulty = deeper search
            rngSeed=ai_rng_seed,
        )
        return HeuristicAI(player, config)

    elif baseline == BaselineOpponent.MCTS_LIGHT:
        # December 2025: Light MCTS (~1500 Elo) - ~32 simulations via think_time
        # MCTS uses think_time (ms) to control search depth, not difficulty
        from app.ai.mcts_ai import MCTSAI
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=difficulty or 4,
            think_time=500,  # 500ms ≈ 32 simulations
            rngSeed=ai_rng_seed,
        )
        return MCTSAI(player, config)

    elif baseline == BaselineOpponent.MCTS_MEDIUM:
        # December 2025: Medium MCTS (~1700 Elo) - ~128 simulations via think_time
        from app.ai.mcts_ai import MCTSAI
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=difficulty or 6,
            think_time=2000,  # 2s ≈ 128 simulations
            rngSeed=ai_rng_seed,
        )
        return MCTSAI(player, config)

    elif baseline == BaselineOpponent.MCTS_STRONG:
        # December 2025: Strong MCTS (~1900 Elo) - ~512 simulations via think_time
        from app.ai.mcts_ai import MCTSAI
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=difficulty or 8,
            think_time=8000,  # 8s ≈ 512 simulations
            rngSeed=ai_rng_seed,
        )
        return MCTSAI(player, config)

    elif baseline == BaselineOpponent.MCTS_MASTER:
        # Dec 28, 2025: Master MCTS (~2000 Elo) - ~1024 simulations via think_time
        from app.ai.mcts_ai import MCTSAI
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=difficulty or 9,
            think_time=16000,  # 16s ≈ 1024 simulations
            rngSeed=ai_rng_seed,
        )
        return MCTSAI(player, config)

    elif baseline == BaselineOpponent.MCTS_GRANDMASTER:
        # Dec 28, 2025: Grandmaster MCTS (~2100 Elo) - ~2048 simulations via think_time
        from app.ai.mcts_ai import MCTSAI
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=difficulty or 10,
            think_time=32000,  # 32s ≈ 2048 simulations
            rngSeed=ai_rng_seed,
        )
        return MCTSAI(player, config)

    # ========================================
    # Dec 29, 2025: NNUE Baseline Opponents
    # These enable unified harness evaluation (Phase 2)
    # ========================================

    elif baseline == BaselineOpponent.NNUE_MINIMAX_D4:
        # NNUE under Minimax depth 4 (~1600 Elo) - best for 2-player games
        from app.ai.minimax_ai import MinimaxAI
        config = AIConfig(
            ai_type=AIType.MINIMAX,
            board_type=board_type,
            difficulty=difficulty or 7,
            use_neural_net=True,  # Enable NNUE evaluation
            rngSeed=ai_rng_seed,
        )
        ai = MinimaxAI(player, config)
        # Override depth to 4 for consistent baseline
        ai._max_depth = 4
        return ai

    elif baseline == BaselineOpponent.NNUE_MAXN_D3:
        # NNUE under MaxN depth 3 (~1650 Elo) - accurate multiplayer search
        if num_players < 3:
            # Fall back to Minimax for 2-player games
            logger.warning(
                f"NNUE_MAXN_D3 is for 3+ players, using NNUE_MINIMAX_D4 for 2-player"
            )
            return create_baseline_ai(
                BaselineOpponent.NNUE_MINIMAX_D4, player, board_type,
                difficulty, game_seed, num_players
            )

        from app.ai.nnue_search_ai import NNUEMaxNAI
        config = AIConfig(
            ai_type=AIType.HEURISTIC,  # Base type, NNUE is injected
            board_type=board_type,
            difficulty=difficulty or 7,
            rngSeed=ai_rng_seed,
        )
        # Jan 13, 2026: Explicitly find NNUE model to avoid silent fallback to heuristic
        nnue_path = _get_nnue_model_path(board_type, num_players)
        if nnue_path is None:
            logger.warning(
                f"[gauntlet] No NNUE model for {board_type}_{num_players}p, MaxN will use heuristic"
            )
        ai = NNUEMaxNAI(
            player_number=player,
            config=config,
            board_type=board_type,
            num_players=num_players,
            model_path=nnue_path,
        )
        return ai

    elif baseline == BaselineOpponent.NNUE_BRS_D3:
        # NNUE under BRS depth 3 (~1550 Elo) - fast multiplayer search
        if num_players < 3:
            # Fall back to Minimax for 2-player games
            logger.warning(
                f"NNUE_BRS_D3 is for 3+ players, using NNUE_MINIMAX_D4 for 2-player"
            )
            return create_baseline_ai(
                BaselineOpponent.NNUE_MINIMAX_D4, player, board_type,
                difficulty, game_seed, num_players
            )

        from app.ai.nnue_search_ai import NNUEBRSAI
        config = AIConfig(
            ai_type=AIType.HEURISTIC,  # Base type, NNUE is injected
            board_type=board_type,
            difficulty=difficulty or 6,
            rngSeed=ai_rng_seed,
        )
        # Jan 13, 2026: Explicitly find NNUE model to avoid silent fallback to heuristic
        nnue_path = _get_nnue_model_path(board_type, num_players)
        if nnue_path is None:
            logger.warning(
                f"[gauntlet] No NNUE model for {board_type}_{num_players}p, BRS will use heuristic"
            )
        ai = NNUEBRSAI(
            player_number=player,
            config=config,
            board_type=board_type,
            num_players=num_players,
            model_path=nnue_path,
        )
        return ai

    # ========================================
    # Dec 29, 2025: Phase 6 - Extended Harness Types
    # PolicyOnly, Gumbel, and Descent variants
    # ========================================

    elif baseline == BaselineOpponent.POLICY_ONLY_NN:
        # PolicyOnly with Full NN (~1200 Elo) - greedy policy selection
        from app.ai.policy_only_ai import PolicyOnlyAI
        config = AIConfig(
            ai_type=AIType.NEURAL_NET,
            board_type=board_type,
            difficulty=difficulty or 5,
            rngSeed=ai_rng_seed,
        )
        return PolicyOnlyAI(player, config, board_type)

    elif baseline == BaselineOpponent.POLICY_ONLY_NNUE:
        # PolicyOnly with NNUE policy (~1100 Elo) - NNUE policy head
        from app.ai.policy_only_ai import PolicyOnlyAI
        config = AIConfig(
            ai_type=AIType.NEURAL_NET,
            board_type=board_type,
            difficulty=difficulty or 5,
            use_nnue_policy=True,  # Enable NNUE policy
            rngSeed=ai_rng_seed,
        )
        return PolicyOnlyAI(player, config, board_type)

    elif baseline == BaselineOpponent.GUMBEL_B64:
        # Gumbel MCTS budget 64 (~1400 Elo) - fast quality search
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        config = AIConfig(
            ai_type=AIType.GUMBEL_MCTS,
            board_type=board_type,
            difficulty=difficulty or 5,
            gumbel_budget=64,
            rngSeed=ai_rng_seed,
        )
        return GumbelMCTSAI(player, config, board_type)

    elif baseline == BaselineOpponent.GUMBEL_B200:
        # Gumbel MCTS budget 200 (~1600 Elo) - balanced quality/speed
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        config = AIConfig(
            ai_type=AIType.GUMBEL_MCTS,
            board_type=board_type,
            difficulty=difficulty or 7,
            gumbel_budget=200,
            rngSeed=ai_rng_seed,
        )
        return GumbelMCTSAI(player, config, board_type)

    elif baseline == BaselineOpponent.GUMBEL_NNUE:
        # Gumbel with NNUE evaluation (~1500 Elo)
        from app.ai.nnue_search_ai import NNUEGumbelAI
        config = AIConfig(
            ai_type=AIType.GUMBEL_MCTS,
            board_type=board_type,
            difficulty=difficulty or 6,
            gumbel_budget=100,
            rngSeed=ai_rng_seed,
        )
        return NNUEGumbelAI(
            player_number=player,
            config=config,
            board_type=board_type,
            num_players=num_players,
        )

    elif baseline == BaselineOpponent.DESCENT_NN:
        # Descent with Full NN (~1400 Elo) - gradient-based search
        from app.ai.descent_ai import DescentAI
        config = AIConfig(
            ai_type=AIType.NEURAL_NET,
            board_type=board_type,
            difficulty=difficulty or 6,
            rngSeed=ai_rng_seed,
        )
        return DescentAI(player, config, board_type)

    elif baseline == BaselineOpponent.DESCENT_NNUE:
        # Descent with NNUE evaluation (~1350 Elo)
        from app.ai.nnue_search_ai import NNUEDescentAI
        config = AIConfig(
            ai_type=AIType.NEURAL_NET,
            board_type=board_type,
            difficulty=difficulty or 6,
            rngSeed=ai_rng_seed,
        )
        return NNUEDescentAI(
            player_number=player,
            config=config,
            board_type=board_type,
            num_players=num_players,
        )

    elif baseline == BaselineOpponent.GPU_GUMBEL:
        # GPU-accelerated Gumbel MCTS (~1600 Elo, 2-10x faster)
        # Uses tensor_gumbel_tree.GPUGumbelMCTS with GPU-batched rollouts
        # Jan 2026: Requires model_path, auto-detects CUDA/MPS/CPU
        from app.ai.harness import HarnessType, create_harness

        # Auto-detect device
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Create GPU Gumbel harness which wraps GPUGumbelMCTS
        # Jan 17, 2026: Scale simulation budget for multiplayer (3p/4p need more sims)
        from app.config.thresholds import get_gauntlet_simulations
        sim_budget = get_gauntlet_simulations(num_players)

        harness = create_harness(
            harness_type=HarnessType.GPU_GUMBEL,
            model_path=model_path,  # Uses canonical model if provided
            board_type=board_type,
            num_players=num_players,
            difficulty=difficulty or 7,
            simulations=sim_budget,  # Scaled: 2p=200, 3p=400, 4p=600
            extra={
                "num_sampled_actions": 16,
                "device": device,
                "eval_mode": "nn",  # Jan 2026: Use NN evaluation for accurate play
            },
        )

        # Return a wrapper that adapts harness.evaluate() to select_move()
        return _HarnessToAIWrapper(harness, player)

    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def create_neural_ai(
    player: int,
    board_type: Any,  # BoardType
    model_path: str | Path | None = None,
    model_getter: Callable[[], Any] | None = None,
    temperature: float = 0.5,
    game_seed: int | None = None,
    num_players: int = 2,
    model_type: str = "cnn",
) -> Any:
    """Create a neural network AI instance.

    Args:
        player: Player number
        board_type: Board type enum
        model_path: Path to model checkpoint (for file-based loading)
        model_getter: Callable that returns model weights (for in-memory loading)
        num_players: Number of players in the game (2, 3, or 4)
        temperature: Policy temperature for move selection
        game_seed: Optional seed for RNG variation per game
        model_type: Type of model - "cnn" (default), "gnn", or "hybrid"

    Returns:
        AI instance (PolicyOnlyAI for CNN, GNNAI for GNN/hybrid)
    """
    _ensure_game_modules()

    # GNN models use dedicated GNNAI class
    if model_type in ("gnn", "hybrid"):
        from app.ai.gnn_ai import create_gnn_ai

        # Derive device - prefer GPU if available
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        try:
            gnn_ai = create_gnn_ai(
                player_number=player,
                model_path=model_path,
                device=device,
                temperature=temperature,
            )
            # Check if model was actually loaded (not None = random fallback)
            if gnn_ai.model is not None:
                return gnn_ai
            # Model is None - checkpoint wasn't GNN format, fall through to CNN
            logger.warning(
                f"GNN model load returned None (checkpoint may be CNN format), "
                f"falling back to UniversalAI for {model_path}"
            )
        except (RuntimeError, ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            # RuntimeError: Model state_dict mismatch, CUDA errors
            # ValueError: Bad model architecture parameters
            # KeyError: Missing keys in checkpoint
            # TypeError: Wrong types in model creation
            # FileNotFoundError/OSError: Checkpoint file not found
            logger.warning(f"GNN AI creation failed: {e}, falling back to UniversalAI")
        # Fall through to CNN path below

    # Derive player-specific seed for varied but reproducible behavior
    ai_rng_seed = None
    if game_seed is not None:
        ai_rng_seed = (game_seed * 104729 + player * 7919) & 0xFFFFFFFF

    if model_path is not None:
        from pathlib import Path
        path_obj = Path(model_path)

        # Try UniversalAI.from_checkpoint first for proper architecture inference
        # This handles checkpoints with different architectures (num_filters, num_res_blocks)
        try:
            # Resolve to absolute path if needed
            if not path_obj.is_absolute():
                # Check common locations
                for prefix in [Path("."), Path("models"), Path("models/checkpoints")]:
                    candidate = prefix / model_path
                    if candidate.exists():
                        path_obj = candidate
                        break
                # Also check with .pth extension
                if not path_obj.exists() and not str(path_obj).endswith(".pth"):
                    for prefix in [Path("."), Path("models"), Path("models/checkpoints")]:
                        candidate = prefix / f"{model_path}.pth"
                        if candidate.exists():
                            path_obj = candidate
                            break

            if path_obj.exists():
                return UniversalAI.from_checkpoint(
                    str(path_obj),
                    player_number=player,
                    board_type=board_type,
                    num_players=num_players,
                    policy_temperature=temperature,
                )
        except (RuntimeError, ValueError, FileNotFoundError, KeyError, OSError) as e:
            logger.warning(f"UniversalAI.from_checkpoint failed: {e}, falling back to PolicyOnlyAI")

        # Fallback to legacy loading via PolicyOnlyAI
        model_id = str(model_path)
        if not path_obj.is_absolute():
            if model_id.startswith("models/"):
                model_id = model_id[7:]
            if model_id.endswith(".pth"):
                model_id = model_id[:-4]

        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_model_id=model_id,
            policy_temperature=temperature,
            rngSeed=ai_rng_seed,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    elif model_getter is not None:
        # In-memory model loading for BackgroundEvaluator (zero disk I/O)
        model_info = model_getter()

        # Extract state_dict from model_info
        if isinstance(model_info, dict):
            if 'state_dict' in model_info:
                state_dict = model_info['state_dict']
            elif 'model_state_dict' in model_info:
                # Versioned checkpoint format
                state_dict = model_info['model_state_dict']
            else:
                # Assume the dict is the state_dict itself
                state_dict = model_info
        elif hasattr(model_info, 'state_dict'):
            # It's an nn.Module - extract state_dict
            state_dict = model_info.state_dict()
        else:
            raise ValueError(
                f"model_getter must return a dict with 'state_dict', a state_dict, "
                f"or an nn.Module. Got: {type(model_info)}"
            )

        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_state_dict=state_dict,
            policy_temperature=temperature,
            rngSeed=ai_rng_seed,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    else:
        raise ValueError("Must provide either model_path or model_getter")


def play_single_game(
    candidate_ai: Any,
    opponent_ai: Any,
    board_type: Any,  # BoardType
    num_players: int = 2,
    candidate_player: int = 1,
    max_moves: int = 500,
    seed: int | None = None,
    opponent_ais: dict[int, Any] | None = None,
    recording_config: Any | None = None,
) -> GameResult:
    """Play a single game between candidate and opponent(s).

    Args:
        candidate_ai: The AI being evaluated
        opponent_ai: The baseline/opponent AI (for 2-player games or fallback)
        board_type: Board type for the game
        num_players: Number of players
        candidate_player: Which player number is the candidate
        max_moves: Maximum moves before draw
        seed: Optional random seed
        opponent_ais: Dict mapping player numbers to AI instances for multiplayer.
                      If not provided, all non-candidate players use opponent_ai.
        recording_config: Optional RecordingConfig for saving game data (Dec 2025).
                         When provided, games are recorded to canonical databases.

    Returns:
        GameResult with game outcome
    """
    _ensure_game_modules()

    engine = DefaultRulesEngine()
    state = create_initial_state(board_type, num_players)

    # Initialize recorder if config provided (Dec 2025 - tournament game recording)
    recorder = None
    if recording_config is not None:
        try:
            from app.db.unified_recording import UnifiedGameRecorder, is_recording_enabled
            if is_recording_enabled():
                import uuid
                recorder = UnifiedGameRecorder(recording_config, state, game_id=str(uuid.uuid4()))
                recorder.__enter__()
        except ImportError:
            pass

    # Build player->AI mapping for multiplayer support
    # NOTE: Players are 1-indexed (1, 2, 3, ...) in the game engine!
    player_ais: dict[int, Any] = {candidate_player: candidate_ai}
    if opponent_ais is not None:
        # Use provided AIs for each opponent player
        player_ais.update(opponent_ais)
    else:
        # For 2-player games, use the single opponent_ai for the other player
        # Players are 1-indexed: range(1, num_players+1) gives [1, 2] for 2-player
        for p in range(1, num_players + 1):
            if p != candidate_player:
                player_ais[p] = opponent_ai

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player

        # Get the AI for the current player
        ai = player_ais.get(current_player)
        if ai is None:
            logger.error(f"No AI assigned for player {current_player}")
            break

        move = ai.select_move(state)

        if move:
            # Extract visit distribution from AI if available (Dec 2025 - Phase 1)
            # Used for richer training signal from MCTS/Gumbel/Descent games
            move_probs: dict[str, float] | None = None
            if hasattr(ai, 'get_visit_distribution'):
                try:
                    moves, probs = ai.get_visit_distribution()
                    if moves and probs:
                        move_probs = {str(m): p for m, p in zip(moves, probs)}
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    pass  # Don't fail on visit distribution extraction error

            # January 9, 2026 (Sprint 17.9): Extract search stats for auxiliary training
            # GumbelMCTSAI provides Q-values, visit counts, search depth, uncertainty
            search_stats: dict | None = None
            if hasattr(ai, 'get_search_stats'):
                try:
                    search_stats = ai.get_search_stats()
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    pass  # Don't fail on search stats extraction error

            state_before = state
            state = engine.apply_move(state, move)

            # Record move if recording enabled (Dec 2025)
            if recorder is not None:
                try:
                    recorder.add_move(
                        move,
                        state_after=state,
                        state_before=state_before,
                        available_moves_count=0,  # Not tracking for gauntlet
                        move_probs=move_probs,  # Dec 2025: Visit distribution
                        search_stats=search_stats,  # Jan 2026: Rich search stats
                    )
                except (RuntimeError, sqlite3.Error, OSError, ValueError):
                    pass  # Don't fail game on recording error
        else:
            # No valid move available - this shouldn't happen in normal games
            logger.warning(f"Player {current_player} returned no move at turn {move_count}")
            break
        move_count += 1

    # Determine outcome
    victory_reason = getattr(state, 'victory_reason', 'unknown')
    if hasattr(victory_reason, 'value'):
        victory_reason = victory_reason.value

    winner = state.winner
    candidate_won = winner == candidate_player if winner is not None else False

    # Finalize recording if enabled (Dec 2025)
    if recorder is not None:
        try:
            recorder.finalize(state, extra_metadata={
                "source": "gauntlet",
                "candidate_player": candidate_player,
                "candidate_won": candidate_won,
                "victory_reason": str(victory_reason) if victory_reason else "max_moves",
            })
            recorder.__exit__(None, None, None)
        except (RuntimeError, sqlite3.Error, OSError, ValueError):
            try:
                recorder.__exit__(None, None, None)
            except (RuntimeError, sqlite3.Error, OSError, ValueError):
                pass

    return GameResult(
        winner=winner,
        move_count=move_count,
        victory_reason=str(victory_reason) if victory_reason else "max_moves",
        candidate_player=candidate_player,
        candidate_won=candidate_won,
    )


def _emit_gauntlet_result_event(
    config_key: str,
    elo: float,
    win_rate: float,
    games: int,
    model_path: str = "",
    vs_random_rate: float | None = None,
    vs_heuristic_rate: float | None = None,
) -> None:
    """Emit EVALUATION_COMPLETED event to close eval→curriculum feedback loop.

    This function emits an event that curriculum_feedback.py's
    TournamentToCurriculumWatcher can consume to adjust training weights.

    Args:
        config_key: Configuration key (e.g., "square8_2p")
        elo: Estimated Elo rating
        win_rate: Overall win rate (0.0-1.0)
        games: Total games played
        model_path: Path to the evaluated model (Jan 2026: for proper Elo tracking)
        vs_random_rate: Win rate vs RANDOM baseline (0.0-1.0)
        vs_heuristic_rate: Win rate vs HEURISTIC baseline (0.0-1.0)
    """
    import asyncio

    try:
        from app.distributed.event_helpers import emit_evaluation_completed_safe

        # Try to emit in async context, otherwise schedule
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(
                emit_evaluation_completed_safe(
                    config=config_key,
                    elo=elo,
                    games=games,
                    win_rate=win_rate,
                    model_path=model_path,  # Jan 2026: Pass model path for Elo tracking
                    source="game_gauntlet",
                    beats_current_best=False,  # Gauntlet is vs baselines, not champion
                    vs_random_rate=vs_random_rate,
                    vs_heuristic_rate=vs_heuristic_rate,
                )
            )
            logger.debug(
                f"[gauntlet] Emitted EVALUATION_COMPLETED for {config_key}: "
                f"elo={elo:.0f}, win_rate={win_rate:.1%}"
            )
        except RuntimeError:
            # No running loop - try to run synchronously
            try:
                asyncio.run(
                    emit_evaluation_completed_safe(
                        config=config_key,
                        elo=elo,
                        games=games,
                        win_rate=win_rate,
                        model_path=model_path,  # Jan 2026: Pass model path for Elo tracking
                        source="game_gauntlet",
                        beats_current_best=False,  # Gauntlet is vs baselines, not champion
                        vs_random_rate=vs_random_rate,
                        vs_heuristic_rate=vs_heuristic_rate,
                    )
                )
            except (RuntimeError, asyncio.CancelledError) as e:
                logger.debug(f"[gauntlet] Could not emit event (no async context): {e}")
    except ImportError:
        logger.debug("[gauntlet] Event helpers not available, skipping event emission")
    except (RuntimeError, OSError, asyncio.CancelledError) as e:
        logger.warning(f"[gauntlet] Failed to emit EVALUATION_COMPLETED: {e}")


def _play_single_gauntlet_game(
    game_num: int,
    baseline: "BaselineOpponent",
    model_path: str | Path | None,
    board_type: Any,
    num_players: int,
    model_getter: Callable[[], Any] | None,
    model_type: str,
    recording_config: Any | None = None,  # Jan 2026: Add recording support for parallel games
) -> dict[str, Any]:
    """Play a single gauntlet game (for parallel execution).

    Args:
        recording_config: Optional RecordingConfig for saving game data (Jan 2026).

    Returns:
        Dict with: game_num, candidate_won, winner, move_count, victory_reason, error
    """
    # Rotate which player the candidate plays as
    candidate_player = (game_num % num_players) + 1
    game_seed = random.randint(0, 0xFFFFFFFF)

    try:
        candidate_ai = create_neural_ai(
            candidate_player, board_type,
            model_path=model_path,
            model_getter=model_getter,
            game_seed=game_seed,
            num_players=num_players,
            model_type=model_type,
        )

        # Create baseline AIs for all other players
        opponent_ais: dict[int, Any] = {}
        for p in range(1, num_players + 1):
            if p != candidate_player:
                opponent_ais[p] = create_baseline_ai(
                    baseline, p, board_type,
                    game_seed=game_seed,
                    num_players=num_players,
                )

        first_opponent = (candidate_player % num_players) + 1
        opponent_ai = opponent_ais.get(first_opponent, list(opponent_ais.values())[0])

        game_result = play_single_game(
            candidate_ai=candidate_ai,
            opponent_ai=opponent_ai,
            board_type=board_type,
            num_players=num_players,
            candidate_player=candidate_player,
            opponent_ais=opponent_ais,
            max_moves=get_theoretical_max_moves(board_type, num_players),
            recording_config=recording_config,  # Jan 2026: Pass through for game recording
        )

        return {
            "game_num": game_num,
            "candidate_won": game_result.candidate_won,
            "winner": game_result.winner,
            "move_count": game_result.move_count,
            "victory_reason": game_result.victory_reason,
            "error": None,
        }
    except (RuntimeError, ValueError, KeyError, OSError, AttributeError) as e:
        return {
            "game_num": game_num,
            "candidate_won": False,
            "winner": None,
            "move_count": 0,
            "victory_reason": "error",
            "error": str(e),
        }


def _evaluate_single_opponent(
    baseline: "BaselineOpponent",
    model_path: str | Path | None,
    board_type: Any,
    games_per_opponent: int,
    num_players: int,
    verbose: bool,
    model_getter: Callable[[], Any] | None,
    model_type: str,
    early_stopping: bool,
    early_stopping_confidence: float,
    early_stopping_min_games: int,
    model_id: str | None = None,
    parallel_games: int | None = None,  # Jan 2026: Default to parallel (get_parallel_games_default())
    recording_config: Any | None = None,
    harness_type: str = "",  # Jan 2026: Harness type for composite Elo tracking
) -> dict[str, Any]:
    """Evaluate a model against a single baseline opponent.

    Args:
        parallel_games: Number of games to run in parallel (default: auto-scaled based on CPU count).
            Jan 2026: Defaults to parallel (16 for 8+ cores, 8 otherwise).
        recording_config: Optional RecordingConfig for saving game data (Dec 2025).

    Returns:
        Dict with keys: baseline_name, wins, games, losses, draws, win_rate,
        early_stopped, games_saved, model_id
    """
    # Jan 2026: Default to parallel execution if not specified
    if parallel_games is None:
        parallel_games = get_parallel_games_default()

    baseline_name = baseline.value

    # Derive model_id if not provided
    effective_model_id = model_id
    if effective_model_id is None and model_path is not None:
        effective_model_id = Path(model_path).stem
    if effective_model_id is None:
        effective_model_id = "candidate_model"

    result = {
        "baseline_name": baseline_name,
        "model_id": effective_model_id,
        "wins": 0,
        "games": 0,
        "losses": 0,
        "draws": 0,
        "win_rate": 0.0,
        "early_stopped": False,
        "games_saved": 0,
    }

    # December 2025 Phase 3: Parallel game execution for faster evaluation
    if parallel_games > 1:
        game_num = 0
        while game_num < games_per_opponent:
            # Determine batch size (don't exceed remaining games)
            batch_size = min(parallel_games, games_per_opponent - game_num)
            batch_games = list(range(game_num, game_num + batch_size))

            # Run batch in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(
                        _play_single_gauntlet_game,
                        g,
                        baseline,
                        model_path,
                        board_type,
                        num_players,
                        model_getter,
                        model_type,
                        recording_config,  # Jan 2026: Pass recording_config for game saving
                    ): g
                    for g in batch_games
                }

                for future in as_completed(futures):
                    try:
                        game_result = future.result()
                        result["games"] += 1

                        if game_result["error"]:
                            logger.debug(f"[gauntlet] Game error: {game_result['error']}")
                            continue

                        if game_result["candidate_won"]:
                            result["wins"] += 1
                        elif game_result["winner"] is not None:
                            result["losses"] += 1
                        else:
                            result["draws"] += 1

                        # Jan 2026: Record Elo for parallel games (was missing, causing empty gauntlet DBs)
                        try:
                            from app.training.elo_recording import (
                                record_gauntlet_match,
                                HarnessType,
                            )

                            board_value = getattr(board_type, "value", str(board_type))
                            harness_enum = HarnessType.from_string(harness_type) if harness_type else HarnessType.GUMBEL_MCTS

                            record_gauntlet_match(
                                model_id=effective_model_id,
                                baseline=baseline_name,
                                model_won=game_result["candidate_won"],
                                board_type=board_value,
                                num_players=num_players,
                                harness_type=harness_enum,
                                game_length=game_result["move_count"],
                            )
                        except ImportError as e:
                            logger.warning(f"[gauntlet] Failed to import elo_recording: {e}")
                        except Exception as e:
                            logger.error(f"[gauntlet] Failed to record gauntlet Elo: {e}")

                        if verbose:
                            outcome = "WIN" if game_result["candidate_won"] else "LOSS"
                            logger.info(
                                f"[gauntlet] Game {game_result['game_num']+1}/{games_per_opponent} "
                                f"vs {baseline_name}: {outcome} ({game_result['move_count']} moves)"
                            )
                    except (RuntimeError, ValueError) as e:
                        logger.error(f"[gauntlet] Parallel game error: {e}")
                        result["games"] += 1

            game_num += batch_size

            # Check early stopping after each batch
            if early_stopping and result["games"] >= early_stopping_min_games:
                if baseline == BaselineOpponent.RANDOM:
                    threshold = get_min_win_rate_vs_random(num_players)
                elif baseline == BaselineOpponent.HEURISTIC:
                    threshold = get_min_win_rate_vs_heuristic(num_players)
                else:
                    threshold = MIN_WIN_RATES.get(baseline, 0.5)

                stop, reason = should_early_stop(
                    wins=result["wins"],
                    losses=result["losses"],
                    threshold=threshold,
                    confidence=early_stopping_confidence,
                    min_games=early_stopping_min_games,
                )

                if stop:
                    games_remaining = games_per_opponent - result["games"]
                    result["early_stopped"] = True
                    result["games_saved"] = games_remaining
                    logger.info(
                        f"[gauntlet] Early stopping vs {baseline_name} at game {result['games']}: "
                        f"{reason} (saved {games_remaining} games)"
                    )
                    break

        # Calculate win rate
        if result["games"] > 0:
            result["win_rate"] = result["wins"] / result["games"]
        return result

    # Sequential execution (original code path)
    for game_num in range(games_per_opponent):
        # Rotate which player the candidate plays as
        # NOTE: Players are 1-indexed (1, 2, ..., num_players) in game engine
        candidate_player = (game_num % num_players) + 1

        # Derive unique seed per game for varied behavior
        game_seed = random.randint(0, 0xFFFFFFFF)

        try:
            candidate_ai = create_neural_ai(
                candidate_player, board_type,
                model_path=model_path,
                model_getter=model_getter,
                game_seed=game_seed,
                num_players=num_players,
                model_type=model_type,
            )

            # Create baseline AIs for all other players
            # NOTE: Players are 1-indexed (1, 2, ..., num_players) in game engine
            opponent_ais: dict[int, Any] = {}
            for p in range(1, num_players + 1):
                if p != candidate_player:
                    opponent_ais[p] = create_baseline_ai(
                        baseline, p, board_type,
                        game_seed=game_seed,
                        num_players=num_players,
                    )

            # For backwards compatibility, also pass opponent_ai (player after candidate)
            first_opponent = (candidate_player % num_players) + 1
            opponent_ai = opponent_ais.get(first_opponent, list(opponent_ais.values())[0])

            game_result = play_single_game(
                candidate_ai=candidate_ai,
                opponent_ai=opponent_ai,
                board_type=board_type,
                num_players=num_players,
                candidate_player=candidate_player,
                opponent_ais=opponent_ais,
                recording_config=recording_config,
                max_moves=get_theoretical_max_moves(board_type, num_players),
            )

            result["games"] += 1

            if game_result.candidate_won:
                result["wins"] += 1
            elif game_result.winner is not None:
                result["losses"] += 1
            else:
                result["draws"] += 1

            # Record match to unified Elo database (December 2025 fix for 3p/4p tracking)
            # January 12, 2026: Migrated to elo_recording facade for better error handling
            try:
                from app.training.elo_recording import (
                    record_gauntlet_match,
                    HarnessType,
                )

                board_value = getattr(board_type, "value", str(board_type))
                model_id = result.get("model_id", "candidate_model")

                # Map harness_type string to HarnessType enum
                harness_enum = HarnessType.from_string(harness_type) if harness_type else HarnessType.GUMBEL_MCTS

                # Use facade - handles composite IDs, validation, error logging, DLQ
                record_gauntlet_match(
                    model_id=model_id,
                    baseline=baseline_name,
                    model_won=game_result.candidate_won,
                    board_type=board_value,
                    num_players=num_players,
                    harness_type=harness_enum,
                    game_length=game_result.move_count,
                )
            except ImportError as e:
                logger.warning(f"[gauntlet] Failed to import elo_recording: {e}")
            except Exception as e:
                logger.error(f"[gauntlet] Failed to record gauntlet Elo: {e}")

            if verbose:
                outcome = "WIN" if game_result.candidate_won else "LOSS"
                logger.info(
                    f"[gauntlet] Game {game_num+1}/{games_per_opponent} vs {baseline_name}: "
                    f"{outcome} ({game_result.victory_reason}, {game_result.move_count} moves)"
                )

            # Emit EVALUATION_PROGRESS event for real-time monitoring (December 2025)
            try:
                from app.coordination.event_router import DataEventType, DataEvent, get_event_bus

                bus = get_event_bus()
                if bus:
                    board_value = getattr(board_type, "value", board_type)
                    if board_value is None:
                        board_value = "unknown"
                    config_key = f"{board_value}_{num_players}p"
                    current_win_rate = result["wins"] / result["games"] if result["games"] > 0 else 0.0
                    bus.publish_sync(DataEvent(
                        event_type=DataEventType.EVALUATION_PROGRESS,
                        payload={
                            "config_key": config_key,
                            "board_type": board_value,
                            "baseline": baseline_name,
                            "games_completed": result["games"],
                            "games_total": games_per_opponent,
                            "wins": result["wins"],
                            "losses": result["losses"],
                            "draws": result["draws"],
                            "current_win_rate": current_win_rate,
                            "num_players": num_players,
                        },
                        source="game_gauntlet",
                    ))
            except (ImportError, AttributeError, RuntimeError):
                pass  # Silent fail - progress events are optional

            # Check for early stopping
            if early_stopping:
                # Get the threshold for this baseline
                if baseline == BaselineOpponent.RANDOM:
                    threshold = get_min_win_rate_vs_random(num_players)
                elif baseline == BaselineOpponent.HEURISTIC:
                    threshold = get_min_win_rate_vs_heuristic(num_players)
                else:
                    threshold = MIN_WIN_RATES.get(baseline, 0.5)

                stop, reason = should_early_stop(
                    wins=result["wins"],
                    losses=result["losses"],
                    threshold=threshold,
                    confidence=early_stopping_confidence,
                    min_games=early_stopping_min_games,
                )

                if stop:
                    games_remaining = games_per_opponent - (game_num + 1)
                    result["early_stopped"] = True
                    result["games_saved"] = games_remaining
                    logger.info(
                        f"[gauntlet] Early stopping vs {baseline_name} at game {game_num+1}: {reason} "
                        f"(saved {games_remaining} games)"
                    )
                    break

        except (RuntimeError, ValueError, KeyError, OSError, AttributeError) as e:
            logger.error(f"Error in game {game_num} vs {baseline_name}: {e}")
            continue

    # Calculate win rate for this opponent
    if result["games"] > 0:
        result["win_rate"] = result["wins"] / result["games"]

    return result


def get_confidence_weighted_games(
    games_per_opponent: int,
    model_id: str | None,
    board_type: str,
    num_players: int,
    max_multiplier: float = 2.0,
) -> int:
    """Calculate adaptive games count based on model's rating confidence.

    New models (low games played, high rating deviation) get more evaluation
    games for better rating accuracy. Established models get fewer games
    since their rating is already well-calibrated.

    Args:
        games_per_opponent: Base games per opponent
        model_id: Model identifier for Elo lookup
        board_type: Board type string
        num_players: Number of players
        max_multiplier: Maximum game count multiplier (default: 2x for new models)

    Returns:
        Adjusted games count based on rating confidence

    December 2025: Added for confidence-weighted evaluation.
    """
    if not model_id:
        return games_per_opponent

    try:
        from app.training.elo_service import get_elo_service
        elo_service = get_elo_service()
        rating = elo_service.get_rating(model_id, board_type, num_players)

        # Calculate confidence from games played
        # 0-30 games: low confidence, 30-100: medium, 100+: high
        games_played = rating.games_played
        if games_played < 30:
            # New model: full multiplier (2x games)
            multiplier = max_multiplier
        elif games_played < 100:
            # Developing model: partial multiplier (1.5x games)
            multiplier = 1.0 + (max_multiplier - 1.0) * (100 - games_played) / 70
        else:
            # Established model: no extra games
            multiplier = 1.0

        adaptive_games = int(games_per_opponent * multiplier)
        if adaptive_games != games_per_opponent:
            logger.debug(
                f"[gauntlet] Confidence-weighted games: {games_per_opponent} -> {adaptive_games} "
                f"(model has {games_played} games)"
            )
        return adaptive_games

    except (ImportError, KeyError, AttributeError, ValueError, TypeError) as e:
        # ImportError: elo_service module not available
        # KeyError: Model not found in Elo database
        # AttributeError: Missing games_played attribute on rating
        # ValueError/TypeError: Invalid parameters or return values
        logger.debug(f"[gauntlet] Could not get rating for confidence weighting: {e}")
        return games_per_opponent


def run_baseline_gauntlet(
    model_path: str | Path | None = None,
    board_type: Any = None,  # BoardType
    opponents: list[BaselineOpponent] | None = None,
    games_per_opponent: int = GAUNTLET_GAMES_PER_OPPONENT,
    num_players: int = 2,
    check_baseline_gating: bool = True,
    verbose: bool = False,
    model_getter: Callable[[], Any] | None = None,
    model_type: str = "cnn",
    store_results: bool = True,
    model_id: str | None = None,
    early_stopping: bool = True,
    early_stopping_confidence: float = 0.95,
    early_stopping_min_games: int = 10,
    elo_adaptive_thresholds: bool = False,
    model_elo: float | None = None,
    parallel_opponents: bool = True,
    max_parallel_workers: int = 2,
    parallel_games: int | None = None,  # Jan 2026: Default to parallel (get_parallel_games_default())
    recording_config: Any | None = None,
    save_games_for_training: bool = True,
    confidence_weighted_games: bool = False,
    harness_type: str = "",  # Dec 29: Harness type for unified evaluation
    game_count: int | None = None,  # Dec 30: Training game count for graduated thresholds
) -> GauntletResult:
    """Run a gauntlet evaluation against baseline opponents.

    Args:
        model_path: Path to the model checkpoint (file-based loading)
        board_type: Board type for games
        opponents: List of baselines to test against (default: RANDOM, HEURISTIC, MCTS_LIGHT, MCTS_MEDIUM)
        games_per_opponent: Number of games per opponent
        num_players: Number of players in each game
        check_baseline_gating: Whether to check minimum win rate thresholds
        verbose: Whether to log per-game results
        model_getter: Callable returning model weights (in-memory loading, zero disk I/O)
        model_type: Type of model - "cnn" (default), "gnn", or "hybrid"
        store_results: Whether to store results in gauntlet_results.db (default: True)
        model_id: Model identifier for result storage (derived from model_path if not specified)
        early_stopping: Whether to stop early when statistical confidence is reached (default: True)
        early_stopping_confidence: Confidence level for early stopping (default: 0.95)
        early_stopping_min_games: Minimum games before early stopping (default: 10)
        elo_adaptive_thresholds: Scale thresholds based on model Elo (default: False)
        model_elo: Model's current Elo rating (required if elo_adaptive_thresholds=True)
        parallel_opponents: Run evaluations against different opponents in parallel (default: True)
            Phase 5 optimization: ~2x speedup when testing vs RANDOM + HEURISTIC concurrently
        max_parallel_workers: Maximum number of parallel opponent evaluations (default: 2)
        parallel_games: Number of games to run in parallel per opponent.
            Jan 2026: Defaults to auto-scaled parallel (16 for 8+ cores, 8 otherwise).
        recording_config: Optional RecordingConfig for saving game data (Dec 2025).
            When provided, games are recorded to canonical databases for training.
        save_games_for_training: Whether to save games for training data (default: True).
            Dec 29, 2025: When True and recording_config is None, creates default config
            that saves games to data/games/gauntlet_{board_type}_{num_players}p.db
        confidence_weighted_games: Whether to adjust games based on model confidence (default: False).
            December 2025: New models (low games played) get up to 2x more evaluation games
            for better rating accuracy. Established models get the base game count.
        harness_type: AI harness type for unified evaluation (e.g., "gumbel_mcts", "minimax").
            December 2025: Enables tracking of model performance under different harnesses.
            Sets harness metadata on result for Elo composite participant IDs.
        game_count: Number of training games available for this config (optional).
            December 30, 2025: When provided, uses graduated thresholds based on data volume:
            - Bootstrap tier (< 5000 games): Lower thresholds for fast iteration
            - Standard tier (5000-20000 games): Normal thresholds
            - Aspirational tier (> 20000 games): Strict thresholds for quality
            This enables model promotion during early training when limited data
            produces weaker models. Takes precedence over elo_adaptive_thresholds.

    Returns:
        GauntletResult with aggregated statistics and harness metadata
    """
    if model_path is None and model_getter is None:
        raise ValueError("Must provide either model_path or model_getter")
    _ensure_game_modules()

    # Jan 2026: Default to parallel execution if not specified
    if parallel_games is None:
        parallel_games = get_parallel_games_default()

    # Dec 30, 2025: Verify model architecture before running gauntlet
    # This catches mismatches (e.g., 4p weights for 2p config) early
    if model_path is not None and board_type is not None:
        is_valid, error_msg = verify_model_architecture(model_path, board_type, num_players)
        if not is_valid:
            logger.error(f"[gauntlet] Architecture verification failed: {error_msg}")
            # Return early with failure result
            result = GauntletResult()
            result.passed = False
            result.model_id = Path(model_path).stem if model_path else None
            result.failure_reason = error_msg
            return result

    # Dec 29, 2025: Create default recording config if saving enabled and not provided
    if save_games_for_training and recording_config is None and board_type is not None:
        recording_config = _create_gauntlet_recording_config(board_type, num_players, source="gauntlet")
        if recording_config is not None:
            logger.info(f"[gauntlet] Saving games to {recording_config.db_path}")

    # Log encoder expectations for debugging channel mismatches
    if board_type is not None:
        try:
            from app.training.encoder_registry import get_encoder_config
            for version in ["v2", "v3"]:
                config = get_encoder_config(board_type, version)
                logger.debug(
                    f"[gauntlet] {board_type.name} {version}: expects {config.in_channels} channels "
                    f"({config.base_channels} base × {config.frames} frames)"
                )
        except (ImportError, AttributeError, KeyError):
            pass  # Registry not available, continue without

    if opponents is None:
        # December 2025: Extended defaults to measure Elo up to ~1800
        # Previous [RANDOM, HEURISTIC] capped observable Elo at ~1200-1500
        opponents = [
            BaselineOpponent.RANDOM,       # ~400 Elo
            BaselineOpponent.HEURISTIC,    # ~1200 Elo
            BaselineOpponent.MCTS_LIGHT,   # ~1500 Elo (32 sims)
            BaselineOpponent.MCTS_MEDIUM,  # ~1700 Elo (128 sims)
        ]

    result = GauntletResult()

    # Dec 29: Set harness metadata on result
    result.harness_type = harness_type
    if harness_type and model_path:
        # Generate composite model ID for Elo tracking
        path_stem = Path(model_path).stem.replace("canonical_", "ringrift_")
        result.model_id = f"{path_stem}:{harness_type}" if harness_type else path_stem
        # Generate config hash if harness abstraction available
        try:
            from app.ai.harness import HarnessType as HT, create_harness
            from app.models import BoardType as BT
            if board_type:
                harness = create_harness(
                    harness_type=HT(harness_type),
                    model_path=model_path,
                    board_type=board_type if isinstance(board_type, BT) else BT(board_type),
                    num_players=num_players,
                )
                result.harness_config_hash = harness.config.get_config_hash()
        except (ImportError, ValueError, KeyError):
            # Harness abstraction not available, use empty hash
            result.harness_config_hash = ""

    # Evaluate opponents (parallel or sequential)
    # Phase 5: Parallel evaluation for ~2x speedup with multiple opponents
    opponent_eval_results: list[dict[str, Any]] = []

    if parallel_opponents and len(opponents) > 1 and not _allow_parallel_opponents():
        logger.debug("[gauntlet] Parallel opponent evaluation disabled (unsafe multiprocessing context)")
        parallel_opponents = False

    # Derive effective_model_id early so it can be passed to evaluators
    effective_model_id = model_id
    if effective_model_id is None and model_path is not None:
        effective_model_id = Path(model_path).stem

    # Apply confidence-weighted games allocation (December 2025)
    effective_games_per_opponent = games_per_opponent
    if confidence_weighted_games and board_type is not None:
        board_type_str = getattr(board_type, "value", str(board_type)).lower()
        effective_games_per_opponent = get_confidence_weighted_games(
            games_per_opponent=games_per_opponent,
            model_id=effective_model_id,
            board_type=board_type_str,
            num_players=num_players,
        )
        if effective_games_per_opponent != games_per_opponent:
            logger.info(
                f"[gauntlet] Using confidence-weighted games: {effective_games_per_opponent} "
                f"(base: {games_per_opponent})"
            )

    if parallel_opponents and len(opponents) > 1:
        # Parallel evaluation using ThreadPoolExecutor
        logger.info(f"[gauntlet] Running parallel evaluation vs {len(opponents)} opponents")
        with ThreadPoolExecutor(max_workers=min(max_parallel_workers, len(opponents))) as executor:
            futures = {
                executor.submit(
                    _evaluate_single_opponent,
                    baseline,
                    model_path,
                    board_type,
                    effective_games_per_opponent,  # Use confidence-weighted games
                    num_players,
                    verbose,
                    model_getter,
                    model_type,
                    early_stopping,
                    early_stopping_confidence,
                    early_stopping_min_games,
                    effective_model_id,
                    parallel_games,
                    recording_config,
                    harness_type,  # Jan 2026: Pass harness type for Elo tracking
                ): baseline
                for baseline in opponents
            }

            for future in as_completed(futures):
                baseline = futures[future]
                try:
                    eval_result = future.result()
                    opponent_eval_results.append(eval_result)
                except (RuntimeError, ValueError, KeyError, OSError, AttributeError) as e:
                    logger.error(f"[gauntlet] Error evaluating vs {baseline.value}: {e}")
                    # Add empty result for this baseline
                    opponent_eval_results.append({
                        "baseline_name": baseline.value,
                        "wins": 0,
                        "games": 0,
                        "losses": 0,
                        "draws": 0,
                        "win_rate": 0.0,
                        "early_stopped": False,
                        "games_saved": 0,
                    })
    else:
        # Sequential evaluation (fallback or single opponent)
        for baseline in opponents:
            eval_result = _evaluate_single_opponent(
                baseline,
                model_path,
                board_type,
                effective_games_per_opponent,  # Use confidence-weighted games
                num_players,
                verbose,
                model_getter,
                model_type,
                early_stopping,
                early_stopping_confidence,
                early_stopping_min_games,
                effective_model_id,
                parallel_games,
                recording_config,
                harness_type,  # Jan 2026: Pass harness type for Elo tracking
            )
            opponent_eval_results.append(eval_result)

    # Aggregate results from all opponent evaluations
    for eval_result in opponent_eval_results:
        baseline_name = eval_result["baseline_name"]
        baseline = BaselineOpponent(baseline_name)

        # Update totals
        result.total_games += eval_result["games"]
        result.total_wins += eval_result["wins"]
        result.total_losses += eval_result["losses"]
        result.total_draws += eval_result["draws"]

        # Track early stopping
        if eval_result["early_stopped"]:
            result.early_stopped_baselines.append(baseline_name)
            result.games_saved_by_early_stopping += eval_result["games_saved"]

        # Store opponent-specific stats
        opponent_stats = {
            "wins": eval_result["wins"],
            "games": eval_result["games"],
            "win_rate": eval_result["win_rate"],
        }
        result.opponent_results[baseline_name] = opponent_stats

        # Check baseline gating (use player-aware or Elo-adaptive thresholds)
        if check_baseline_gating:
            # Use Elo-adaptive thresholds if enabled and model_elo provided
            # Dec 30, 2025: Priority order for threshold selection:
            # 1. Game count graduated thresholds (if game_count provided)
            # 2. Elo-adaptive thresholds (if elo_adaptive_thresholds and model_elo)
            # 3. Static player-aware thresholds (fallback)
            if game_count is not None:
                # Use game count-based graduated thresholds
                tier = get_game_count_tier(game_count)
                if baseline == BaselineOpponent.RANDOM:
                    min_required = get_game_count_based_win_rate_vs_random(game_count, num_players)
                elif baseline == BaselineOpponent.HEURISTIC:
                    min_required = get_game_count_based_win_rate_vs_heuristic(game_count, num_players)
                else:
                    min_required = MIN_WIN_RATES.get(baseline, 0.0)
                threshold_type = f"{num_players}p/{tier}@{game_count}games"
            elif elo_adaptive_thresholds and model_elo is not None:
                if baseline == BaselineOpponent.RANDOM:
                    min_required = get_elo_adaptive_win_rate_vs_random(model_elo, num_players)
                elif baseline == BaselineOpponent.HEURISTIC:
                    min_required = get_elo_adaptive_win_rate_vs_heuristic(model_elo, num_players)
                else:
                    min_required = MIN_WIN_RATES.get(baseline, 0.0)
                threshold_type = f"{num_players}p/elo-adaptive@{model_elo:.0f}"
            else:
                # Fallback to static player-aware thresholds
                if baseline == BaselineOpponent.RANDOM:
                    min_required = get_min_win_rate_vs_random(num_players)
                elif baseline == BaselineOpponent.HEURISTIC:
                    min_required = get_min_win_rate_vs_heuristic(num_players)
                else:
                    min_required = MIN_WIN_RATES.get(baseline, 0.0)
                threshold_type = f"{num_players}p thresholds"

            # Populate thresholds_met for this baseline (January 2026 fix)
            threshold_passed = opponent_stats["win_rate"] >= min_required
            result.thresholds_met[baseline_name] = threshold_passed

            if not threshold_passed:
                result.passes_baseline_gating = False
                result.failed_baselines.append(baseline_name)
                logger.warning(
                    f"[gauntlet] Failed baseline gating vs {baseline_name}: "
                    f"{opponent_stats['win_rate']:.1%} < {min_required:.0%} required"
                    f" ({threshold_type})"
                )

    # Calculate overall win rate
    if result.total_games > 0:
        result.win_rate = result.total_wins / result.total_games

    # Estimate Elo from win rates
    result.estimated_elo = _estimate_elo_from_results(result.opponent_results)

    # Log comprehensive summary (January 2026)
    logger.info("=" * 60)
    logger.info(f"GAUNTLET SUMMARY: {effective_model_id}")
    logger.info("=" * 60)
    for opp_name, opp_stats in result.opponent_results.items():
        wins = opp_stats.get("wins", 0)
        total = opp_stats.get("games", 0)
        pct = (wins / total * 100) if total > 0 else 0.0
        logger.info(f"  {opp_name}: {wins}/{total} ({pct:.1f}%)")
    logger.info(f"  Overall: {result.total_wins}/{result.total_games} ({result.win_rate*100:.1f}%)")
    logger.info(f"  Estimated Elo: {result.estimated_elo:.0f}")
    logger.info("=" * 60)

    # Emit EVALUATION_COMPLETED event for curriculum feedback (December 2025)
    # This closes the eval→curriculum feedback loop
    board_type_str = board_type.value if hasattr(board_type, "value") else str(board_type)

    # Extract baseline-specific win rates for promotion daemon (Dec 28, 2025)
    random_stats = result.opponent_results.get("random", {})
    heuristic_stats = result.opponent_results.get("heuristic", {})
    vs_random_rate = random_stats.get("win_rate") if random_stats else None
    vs_heuristic_rate = heuristic_stats.get("win_rate") if heuristic_stats else None

    # Jan 2026: Pass model_path to prevent config-as-model_id bug in Elo tracking
    model_path_str = str(model_path) if model_path else ""
    _emit_gauntlet_result_event(
        config_key=f"{board_type_str}_{num_players}p",
        elo=result.estimated_elo,
        win_rate=result.win_rate,
        games=result.total_games,
        model_path=model_path_str,
        vs_random_rate=vs_random_rate,
        vs_heuristic_rate=vs_heuristic_rate,
    )

    # Store results in gauntlet_results.db (December 2025)
    if store_results:
        try:
            from app.training.gauntlet_results_db import get_gauntlet_db

            # Derive model_id from path if not specified
            effective_model_id = model_id
            if effective_model_id is None and model_path is not None:
                effective_model_id = Path(model_path).stem

            if effective_model_id:
                db = get_gauntlet_db()
                for baseline_name, stats in result.opponent_results.items():
                    db.store_result(
                        model_id=effective_model_id,
                        board_type=board_type.value if hasattr(board_type, "value") else str(board_type),
                        num_players=num_players,
                        opponent=baseline_name,
                        wins=stats.get("wins", 0),
                        losses=stats.get("games", 0) - stats.get("wins", 0),
                        draws=0,
                        metadata={
                            "estimated_elo": result.estimated_elo,
                            "model_type": model_type,
                            "passes_gating": result.passes_baseline_gating,
                            # Dec 29: Harness metadata for unified evaluation
                            "harness_type": result.harness_type or "",
                            "harness_config_hash": result.harness_config_hash or "",
                            "model_id": result.model_id or "",
                        },
                    )
                logger.debug(f"[gauntlet] Stored results for {effective_model_id} in gauntlet_results.db")
        except Exception as e:
            # Jan 12, 2026: Widened from (OSError, RuntimeError, ValueError) to catch sqlite3 errors,
            # TypeErrors, AttributeErrors, etc. that were silently dropped and caused empty gauntlet_results.
            logger.warning(f"[gauntlet] Failed to store results in DB: {type(e).__name__}: {e}")

    # Clear model cache after gauntlet to release GPU/MPS memory (Dec 29, 2025)
    if HAS_MODEL_CACHE and clear_model_cache is not None:
        clear_model_cache()
        logger.debug("[gauntlet] Cleared model cache after gauntlet evaluation")

    # January 2026: Emit batch event to trigger consolidation pipeline
    # This ensures gauntlet games are consolidated into canonical databases for training
    if recording_config is not None and result.total_games > 0:
        try:
            from app.db.unified_recording import emit_games_recorded_batch
            emit_games_recorded_batch(recording_config, result.total_games)
            logger.debug(
                f"[gauntlet] Emitted NEW_GAMES_AVAILABLE for {result.total_games} games "
                f"({board_type_str}_{num_players}p)"
            )
        except ImportError:
            pass

    return result


def _estimate_elo_from_results(
    opponent_results: dict[str, dict[str, Any]]
) -> float:
    """Estimate Elo rating from gauntlet results.

    Uses weighted average of Elo estimates from each opponent.
    """
    import math

    total_elo = 0.0
    total_weight = 0.0

    for baseline_name, stats in opponent_results.items():
        baseline = BaselineOpponent(baseline_name)
        opponent_elo = BASELINE_ELOS.get(baseline, 1000)
        win_rate = stats.get("win_rate", 0.5)
        games = stats.get("games", 0)

        if games == 0:
            continue

        # Elo formula: E = 1 / (1 + 10^((Rb-Ra)/400))
        # Solving for Ra: Ra = Rb - 400 * log10(1/E - 1)
        if win_rate <= 0:
            estimated = opponent_elo - 400
        elif win_rate >= 1:
            estimated = opponent_elo + 400
        else:
            estimated = opponent_elo - 400 * math.log10(1/win_rate - 1)

        total_elo += estimated * games
        total_weight += games

    if total_weight > 0:
        return total_elo / total_weight
    return 1500.0


# ============================================
# Baseline Calibration (Dec 2025)
# ============================================
# Run baseline-vs-baseline games to calibrate Elo ratings


async def run_baseline_calibration(
    board_type: Any,
    num_players: int,
    games_per_matchup: int = 50,
    save_games: bool = True,
) -> dict[str, Any]:
    """Run baseline-vs-baseline games to calibrate Elo ratings.

    This ensures heuristic vs random Elo is properly calibrated by playing
    them against each other directly, not just through NN matches.

    Dec 29, 2025: Added to ensure diverse Elo population with all baseline
    matchups including heuristic vs random.

    Args:
        board_type: Board type for games
        num_players: Number of players in each game
        games_per_matchup: Number of games per baseline pair (default: 50)
        save_games: Whether to save games for training (default: True)

    Returns:
        Dict with win rates for each baseline pair
    """
    _ensure_game_modules()

    # January 12, 2026: Use elo_recording facade for better error handling
    try:
        from app.training.elo_recording import record_baseline_calibration_match
        has_elo_recording = True
    except ImportError:
        logger.warning("[calibration] elo_recording facade not available, skipping Elo recording")
        has_elo_recording = False

    results: dict[str, Any] = {}
    board_value = getattr(board_type, "value", str(board_type))

    # Create recording config if saving games
    recording_config = None
    if save_games:
        recording_config = _create_gauntlet_recording_config(
            board_type, num_players, source="baseline_calibration"
        )

    # Baseline pairs to calibrate (weaker vs stronger)
    baseline_pairs = [
        (BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC),
        (BaselineOpponent.HEURISTIC, BaselineOpponent.MCTS_LIGHT),
    ]

    for baseline_a, baseline_b in baseline_pairs:
        matchup_key = f"{baseline_a.value}_vs_{baseline_b.value}"
        logger.info(f"[calibration] Running {matchup_key} ({games_per_matchup} games)")

        wins_a, wins_b, draws = 0, 0, 0

        for game_idx in range(games_per_matchup):
            # Alternate who plays first for fairness
            if game_idx % 2 == 0:
                player_a, player_b = 1, 2
            else:
                player_a, player_b = 2, 1

            game_seed = random.randint(0, 0xFFFFFFFF)

            ai_a = create_baseline_ai(
                baseline_a, player_a, board_type,
                game_seed=game_seed, num_players=num_players
            )
            ai_b = create_baseline_ai(
                baseline_b, player_b, board_type,
                game_seed=game_seed, num_players=num_players
            )

            # Build player_ais dict for multiplayer
            player_ais = {player_a: ai_a, player_b: ai_b}

            # Fill remaining players with random AI for 3p/4p games
            for p in range(1, num_players + 1):
                if p not in player_ais:
                    player_ais[p] = create_baseline_ai(
                        BaselineOpponent.RANDOM, p, board_type,
                        game_seed=game_seed, num_players=num_players
                    )

            # Play the game
            game_result = play_single_game(
                candidate_ai=ai_a,
                opponent_ai=ai_b,
                board_type=board_type,
                num_players=num_players,
                candidate_player=player_a,
                opponent_ais={p: ai for p, ai in player_ais.items() if p != player_a},
                recording_config=recording_config,
            )

            # Track result
            if game_result.winner == player_a:
                wins_a += 1
                winner_baseline = baseline_a.value
            elif game_result.winner == player_b:
                wins_b += 1
                winner_baseline = baseline_b.value
            else:
                draws += 1
                winner_baseline = None

            # Record match for Elo update using facade
            # January 12, 2026: Migrated to elo_recording facade - handles errors internally
            if has_elo_recording:
                record_baseline_calibration_match(
                    baseline_a=baseline_a.value,
                    baseline_b=baseline_b.value,
                    winner=winner_baseline,
                    board_type=board_value,
                    num_players=num_players,
                    game_length=game_result.move_count,
                )

        # Store results for this matchup
        total = wins_a + wins_b + draws
        results[matchup_key] = {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "games": total,
            "win_rate_a": wins_a / total if total > 0 else 0.0,
            "win_rate_b": wins_b / total if total > 0 else 0.0,
        }

        logger.info(
            f"[calibration] {matchup_key}: {baseline_a.value} {wins_a}-{wins_b} {baseline_b.value} "
            f"({draws} draws)"
        )

    return results


def run_baseline_calibration_sync(
    board_type: Any,
    num_players: int,
    games_per_matchup: int = 50,
    save_games: bool = True,
) -> dict[str, Any]:
    """Synchronous wrapper for run_baseline_calibration().

    Use this for CLI scripts or non-async contexts.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # Already in async context - create task
        return asyncio.ensure_future(
            run_baseline_calibration(board_type, num_players, games_per_matchup, save_games)
        )
    except RuntimeError:
        # No running loop - run synchronously
        return asyncio.run(
            run_baseline_calibration(board_type, num_players, games_per_matchup, save_games)
        )


# Convenience function for quick evaluation
def quick_evaluate(
    model_path: str | Path,
    games: int = 10,
) -> dict[str, float]:
    """Quick evaluation against baselines.

    Args:
        model_path: Path to model checkpoint
        games: Games per opponent

    Returns:
        Dict with win rates per opponent
    """
    _ensure_game_modules()

    result = run_baseline_gauntlet(
        model_path=model_path,
        board_type=BoardType.SQUARE8,
        games_per_opponent=games,
        verbose=True,
    )

    return {
        name: stats["win_rate"]
        for name, stats in result.opponent_results.items()
    }


def run_model_vs_model(
    model_a_path: str | Path,
    model_b_path: str | Path,
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 50,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run head-to-head games between two neural network models.

    January 10, 2026: Added for promotion gate - ensures new models beat current canonical.

    Args:
        model_a_path: Path to first model (the candidate)
        model_b_path: Path to second model (the baseline/canonical)
        board_type: Board type for games
        num_players: Number of players
        num_games: Number of games to play (alternates who goes first)
        verbose: Print progress

    Returns:
        Dict with:
            - win_rate: Win rate of model_a vs model_b
            - games_played: Total games played
            - wins: Number of wins for model_a
            - losses: Number of losses for model_a
            - draws: Number of draws
    """
    _ensure_game_modules()

    from app.ai.mcts_ai import MCTSAI
    from app.rules.game_state import GameState
    from app.rules.game_engine import GameEngine

    # Load models
    if verbose:
        logger.info(f"Loading model A: {model_a_path}")
        logger.info(f"Loading model B: {model_b_path}")

    # Create AI instances with the models
    # Jan 17, 2026: Scale simulation budget for multiplayer (3p/4p need more sims)
    from app.config.thresholds import get_gauntlet_simulations
    sim_budget = get_gauntlet_simulations(num_players)

    ai_a = MCTSAI(
        str(model_a_path),
        simulations=sim_budget,
        board_type=board_type,
        num_players=num_players,
    )
    ai_b = MCTSAI(
        str(model_b_path),
        simulations=sim_budget,
        board_type=board_type,
        num_players=num_players,
    )

    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(num_games):
        # Alternate who goes first
        if game_idx % 2 == 0:
            players = [ai_a, ai_b]
            a_is_player = 0
        else:
            players = [ai_b, ai_a]
            a_is_player = 1

        # Initialize game
        state = GameState.create(board_type=board_type, num_players=num_players)
        max_moves = get_theoretical_max_moves(board_type, num_players)
        move_count = 0

        # Play game
        while not state.game_over and move_count < max_moves:
            current_player = state.current_player
            ai = players[current_player]
            move = ai.get_move(state)
            if move is None:
                break
            state = GameEngine.apply_move(state, move)
            move_count += 1

        # Determine winner
        if state.game_over and state.winner is not None:
            if state.winner == a_is_player:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

        if verbose and (game_idx + 1) % 10 == 0:
            logger.info(
                f"Head-to-head progress: {game_idx + 1}/{num_games} "
                f"(A wins: {wins}, B wins: {losses}, draws: {draws})"
            )

    total_games = wins + losses + draws
    win_rate = wins / total_games if total_games > 0 else 0.0

    if verbose:
        logger.info(
            f"Head-to-head complete: {wins}/{total_games} wins ({win_rate:.1%})"
        )

    return {
        "win_rate": win_rate,
        "games_played": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }
