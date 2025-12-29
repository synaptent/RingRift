"""Unified Threshold Constants for RingRift AI Service.

This module provides SINGLE SOURCE OF TRUTH for all threshold values
used across training, evaluation, promotion, and rollback systems.

Import these constants instead of hardcoding values:

    from app.config.thresholds import (
        TRAINING_TRIGGER_GAMES,
        ELO_DROP_ROLLBACK,
        ELO_IMPROVEMENT_PROMOTE,
    )

See ai-service/docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

from functools import lru_cache

# =============================================================================
# Training Thresholds
# =============================================================================

# Games needed to trigger training (per config)
# December 29, 2025: Reduced from 100 to 50 for faster pipeline iteration
TRAINING_TRIGGER_GAMES = 50

# Minimum interval between training runs (seconds)
# December 29, 2025: Reduced from 1200 to 300 for faster iteration cycles
TRAINING_MIN_INTERVAL_SECONDS = 300  # 5 minutes

# Hours before a config is considered "stale" (no recent training)
TRAINING_STALENESS_HOURS = 6.0

# Bootstrap threshold for new configs (0 models)
TRAINING_BOOTSTRAP_GAMES = 50

# Maximum concurrent training jobs
TRAINING_MAX_CONCURRENT = 3

# =============================================================================
# Regression & Rollback Thresholds
# =============================================================================

# Elo drop that triggers rollback consideration
ELO_DROP_ROLLBACK = 50

# Win rate drop percentage that triggers rollback
WIN_RATE_DROP_ROLLBACK = 0.10  # 10%

# Error rate threshold for rollback
ERROR_RATE_ROLLBACK = 0.05  # 5%

# Minimum games for reliable regression detection
MIN_GAMES_REGRESSION = 50

# Consecutive regressions before forced rollback
CONSECUTIVE_REGRESSIONS_FORCE = 3

# =============================================================================
# Promotion Thresholds
# =============================================================================

# Elo improvement required for promotion
ELO_IMPROVEMENT_PROMOTE = 20

# Minimum games before eligible for promotion
MIN_GAMES_PROMOTE = 100

# Minimum win rate for promotion consideration
MIN_WIN_RATE_PROMOTE = 0.45

# Win rate required to beat current best in head-to-head tournament
# (candidate must demonstrate superiority with >55% decisive wins)
WIN_RATE_BEAT_BEST = 0.55

# =============================================================================
# Tier Ladder Evaluation Thresholds (December 2025)
# =============================================================================
# D(n) must beat D(n-1) at this win rate on BOTH sq8 and hex8 2p matches

# Canonical tier-vs-tier threshold for ladder promotion
# D(n) must beat D(n-1) at 55%+ in 2-player matches
TIER_VS_PREVIOUS_MIN_WIN_RATE = 0.55

# Board types required for tier validation (must pass on ALL)
TIER_VALIDATION_BOARD_TYPES = ["square8", "hex8"]

# Number of players for tier validation
TIER_VALIDATION_NUM_PLAYERS = 2

# Games per board type for tier evaluation
TIER_VALIDATION_GAMES_PER_BOARD = 100

# Games per opponent for gauntlet evaluation (increased from 20 for better precision)
# At 50 games: ~14% confidence interval on win rate
# At 100 games: ~10% confidence interval on win rate
GAUNTLET_GAMES_PER_OPPONENT = 50

# Dec 29, 2025: Multiplayer gauntlet games (higher due to variance)
# 3p/4p games have higher variance (33%/25% random baseline vs 50%)
# requiring more games for statistical significance
GAUNTLET_GAMES_PER_OPPONENT_3P = 75   # 3-player: 50% more games
GAUNTLET_GAMES_PER_OPPONENT_4P = 100  # 4-player: 100% more games


def get_gauntlet_games_per_opponent(num_players: int = 2) -> int:
    """Get recommended gauntlet games per opponent based on player count.

    Higher player counts have higher variance (33% vs 50% vs 25% random baseline),
    requiring more games for statistical significance.

    Args:
        num_players: Number of players (2, 3, or 4)

    Returns:
        Recommended number of gauntlet games per opponent
    """
    if num_players >= 4:
        return GAUNTLET_GAMES_PER_OPPONENT_4P
    if num_players == 3:
        return GAUNTLET_GAMES_PER_OPPONENT_3P
    return GAUNTLET_GAMES_PER_OPPONENT


# Dec 29, 2025: Minimum games for export by player count
# Higher player counts have higher variance and need more data
# December 29, 2025: Reduced thresholds for faster pipeline iteration
MIN_GAMES_FOR_EXPORT_2P = 200   # 2-player baseline (was 500)
MIN_GAMES_FOR_EXPORT_3P = 300   # 3-player (was 600)
MIN_GAMES_FOR_EXPORT_4P = 400   # 4-player (was 800)


def get_min_games_for_export(num_players: int = 2) -> int:
    """Get minimum games needed for export based on player count.

    Higher player counts have higher variance (multiple winners possible,
    more complex game states), requiring more training data.

    Dec 29, 2025: Added as part of Phase 2 training loop improvements.

    Args:
        num_players: Number of players (2, 3, or 4)

    Returns:
        Minimum number of games before triggering export
    """
    if num_players >= 4:
        return MIN_GAMES_FOR_EXPORT_4P
    if num_players == 3:
        return MIN_GAMES_FOR_EXPORT_3P
    return MIN_GAMES_FOR_EXPORT_2P


# Cooldown between promotion attempts (seconds)
PROMOTION_COOLDOWN_SECONDS = 900  # 15 minutes

# =============================================================================
# Production Promotion Thresholds (December 2025)
# =============================================================================
# These are the hard gates for promoting a model to production use

# Minimum ELO rating for production promotion
# Source of truth: tests expect production to align with the "expert" tier.
# (Random≈400, Heuristic≈1200, Expert/Production=1650)
PRODUCTION_ELO_THRESHOLD = 1650

# Minimum games played before production promotion eligible
# Rationale: 100 games gives ~10% confidence interval on ELO
PRODUCTION_MIN_GAMES = 100

# Minimum win rate vs heuristic for production
# Rationale: Must consistently beat heuristic to be useful
PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC = 0.60

# Minimum win rate vs random for production
# Rationale: Must not have degenerate behavior
PRODUCTION_MIN_WIN_RATE_VS_RANDOM = 0.90

# ELO rating tiers for reporting
ELO_TIER_NOVICE = 800       # Below heuristic
ELO_TIER_INTERMEDIATE = 1200  # Heuristic-level
ELO_TIER_ADVANCED = 1500    # Better than heuristic
ELO_TIER_EXPERT = 1650      # Production-ready
ELO_TIER_MASTER = 1800      # Strong model
ELO_TIER_GRANDMASTER = 2000  # Exceptional model

# Archive threshold for tournament management
# Models below this ELO (with sufficient games) may be archived from active tournaments
# Rationale: ~1400 is around heuristic-level, models below are likely not improving
ARCHIVE_ELO_THRESHOLD = 1400

# =============================================================================
# Absolute Elo Targets Per Configuration (December 2025)
# =============================================================================
# Target 2000+ Elo for all 12 board/player combinations
# This drives training priority - configs further from target get more resources

# Global target - all configurations should achieve this Elo
ELO_TARGET_ALL_CONFIGS = 2000.0

# Per-configuration targets (can be customized if some combos are harder)
ELO_TARGETS_BY_CONFIG: dict[str, float] = {
    # Square 8x8 board
    "square8_2p": 2000.0,
    "square8_3p": 2000.0,
    "square8_4p": 2000.0,
    # Square 19x19 board
    "square19_2p": 2000.0,
    "square19_3p": 2000.0,
    "square19_4p": 2000.0,
    # Hex 8 board
    "hex8_2p": 2000.0,
    "hex8_3p": 2000.0,
    "hex8_4p": 2000.0,
    # Hexagonal (11) board
    "hexagonal_2p": 2000.0,
    "hexagonal_3p": 2000.0,
    "hexagonal_4p": 2000.0,
}

# All 12 configuration keys for iteration
ALL_CONFIG_KEYS = list(ELO_TARGETS_BY_CONFIG.keys())

# Minimum Elo threshold to consider a config "production-ready"
# Models must exceed this AND relative improvement thresholds
ELO_PRODUCTION_GATE = 2000.0


def get_elo_target(config_key: str) -> float:
    """Get the Elo target for a specific configuration.

    Args:
        config_key: Configuration key like 'square8_2p', 'hexagonal_4p'

    Returns:
        Target Elo for this configuration (default: 2000.0)
    """
    return ELO_TARGETS_BY_CONFIG.get(config_key, ELO_TARGET_ALL_CONFIGS)


def get_elo_gap(config_key: str, current_elo: float) -> float:
    """Calculate the gap between current Elo and target.

    Args:
        config_key: Configuration key
        current_elo: Current best Elo for this config

    Returns:
        Positive gap (target - current), or 0 if target met
    """
    target = get_elo_target(config_key)
    return max(0.0, target - current_elo)


def is_target_met(config_key: str, current_elo: float) -> bool:
    """Check if a configuration has met its Elo target.

    Args:
        config_key: Configuration key
        current_elo: Current best Elo for this config

    Returns:
        True if current_elo >= target
    """
    return current_elo >= get_elo_target(config_key)


def get_priority_weight(config_key: str, current_elo: float) -> float:
    """Calculate training priority weight based on Elo gap.

    Configs further from target get higher priority (more training resources).
    Uses quadratic scaling so large gaps get exponentially more attention.

    Args:
        config_key: Configuration key
        current_elo: Current best Elo for this config

    Returns:
        Priority weight (1.0 = baseline, higher = more urgent)
    """
    if is_target_met(config_key, current_elo):
        return 0.1  # Maintenance mode for configs at target

    gap = get_elo_gap(config_key, current_elo)
    # Quadratic scaling: 100 gap = 2x, 200 gap = 5x, 300 gap = 10x
    return min(10.0, 1.0 + (gap / 100) ** 1.5)

# =============================================================================
# Elo Rating System
# =============================================================================

# Initial Elo rating for new models/players
INITIAL_ELO_RATING = 1500.0

# Minimum Elo rating (floor)
MIN_ELO_RATING = 100.0

# Maximum Elo rating (ceiling)
MAX_ELO_RATING = 3000.0

# =============================================================================
# Baseline Gating Thresholds
# =============================================================================
# Checkpoints must beat baselines at these rates to be considered "qualified"
# This prevents selecting checkpoints strong in neural-vs-neural but weak vs basics

# Minimum win rate against random AI for checkpoint qualification
# Dec 28, 2025: Raised from 70% to 85% - weak data causes Elo plateau
MIN_WIN_RATE_VS_RANDOM = 0.85  # 85% (must dominate random)

# Minimum win rate against heuristic AI for checkpoint qualification
# Dec 28, 2025: Raised from 75% to 85% - 75% still plateaus at ~1650 Elo.
# AlphaZero-quality training requires decisively beating heuristics.
# 85% vs heuristic correlates with 1800+ Elo potential.
MIN_WIN_RATE_VS_HEURISTIC = 0.85  # 85% (must decisively beat heuristic for 1800+ Elo)

# -----------------------------------------------------------------------------
# 3-Player Adjusted Thresholds
# -----------------------------------------------------------------------------
# For 3-player games, random baseline is 33% (1/3 chance), not 50%.
# Dec 28, 2025: Raised thresholds - weak data was causing plateau
# 85% for 2p (2.55x over 33%) -> 55% for 3p (1.65x over 33%)
# Heuristic raised to 45% to ensure quality training data

MIN_WIN_RATE_VS_RANDOM_3P = 0.55  # 55% (1.65x better than 33% random baseline)
MIN_WIN_RATE_VS_HEURISTIC_3P = 0.45  # 45% (must beat heuristic convincingly)

# -----------------------------------------------------------------------------
# 4-Player Adjusted Thresholds
# -----------------------------------------------------------------------------
# For 4-player games, random baseline is 25% (1/4 chance), not 50%.
# Dec 29, 2025: Fixed thresholds - 4p should be LOWER than 3p, not higher!
# Using consistent ~1.7x multiplier over random baseline:
#   2p: 50% * 1.7 = 85%
#   3p: 33% * 1.7 = 56% ~ 55%
#   4p: 25% * 1.7 = 42.5% ~ 45%

MIN_WIN_RATE_VS_RANDOM_4P = 0.45  # 45% (1.8x better than 25% random baseline)
MIN_WIN_RATE_VS_HEURISTIC_4P = 0.35  # 35% (scaled down from 3p's 45%)


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


# =============================================================================
# Elo-Adaptive Thresholds (December 2025)
# =============================================================================
# Scale promotion requirements based on model strength. Early training models
# have easier thresholds to bootstrap quickly; strong models have strict gates.


def get_elo_adaptive_win_rate_vs_random(model_elo: float, num_players: int = 2) -> float:
    """Get Elo-adaptive win rate threshold vs random baseline.

    Thresholds scale with model strength:
    - Weak models (< 1300 Elo): Lower threshold for fast iteration
    - Medium models (1300-1500): Standard threshold
    - Strong models (1500-1700): Higher threshold for quality
    - Very strong (> 1700): Strict threshold

    Args:
        model_elo: Current model Elo rating
        num_players: Number of players (2, 3, or 4)

    Returns:
        Adaptive minimum win rate vs random
    """
    # Get base threshold from player count
    base = get_min_win_rate_vs_random(num_players)

    # Scale based on Elo
    if model_elo < 1300:
        # Weak model: easier threshold (0.85x of base)
        multiplier = 0.85
    elif model_elo < 1500:
        # Medium model: standard threshold
        multiplier = 1.0
    elif model_elo < 1700:
        # Strong model: stricter threshold (1.1x of base)
        multiplier = 1.1
    else:
        # Very strong: strictest threshold (1.2x of base)
        multiplier = 1.2

    # Clamp to [0.5, 0.95] range
    return min(0.95, max(0.5, base * multiplier))


def get_elo_adaptive_win_rate_vs_heuristic(model_elo: float, num_players: int = 2) -> float:
    """Get Elo-adaptive win rate threshold vs heuristic baseline.

    Thresholds scale with model strength:
    - Weak models (< 1300 Elo): Lower threshold (break-even is fine)
    - Medium models (1300-1500): Standard threshold
    - Strong models (1500-1700): Higher threshold
    - Very strong (> 1700): Strict threshold (must dominate heuristic)

    Args:
        model_elo: Current model Elo rating
        num_players: Number of players (2, 3, or 4)

    Returns:
        Adaptive minimum win rate vs heuristic
    """
    # Get base threshold from player count
    base = get_min_win_rate_vs_heuristic(num_players)

    # Scale based on Elo
    if model_elo < 1300:
        # Weak model: easier threshold (0.8x of base)
        multiplier = 0.8
    elif model_elo < 1500:
        # Medium model: standard threshold
        multiplier = 1.0
    elif model_elo < 1700:
        # Strong model: stricter threshold (1.15x of base)
        multiplier = 1.15
    else:
        # Very strong: strictest threshold (1.3x of base)
        multiplier = 1.3

    # Clamp to [0.15, 0.85] range (heuristic is tough)
    return min(0.85, max(0.15, base * multiplier))


def get_adaptive_thresholds(model_elo: float, num_players: int = 2) -> dict[str, float]:
    """Get all adaptive thresholds for a model based on Elo.

    Convenience function that returns both random and heuristic thresholds.

    Args:
        model_elo: Current model Elo rating
        num_players: Number of players

    Returns:
        Dict with 'random' and 'heuristic' win rate thresholds
    """
    return {
        "random": get_elo_adaptive_win_rate_vs_random(model_elo, num_players),
        "heuristic": get_elo_adaptive_win_rate_vs_heuristic(model_elo, num_players),
    }


# Baseline Elo estimates for Elo calculation from win rates
BASELINE_ELO_RANDOM = 400
BASELINE_ELO_HEURISTIC = 1200

# December 2025: Extended baseline Elo ladder for measuring higher Elo models
# These enable accurate Elo estimation above the previous 1600 ceiling
BASELINE_ELO_HEURISTIC_STRONG = 1400  # Heuristic at difficulty 8
BASELINE_ELO_MCTS_LIGHT = 1500        # MCTS with 32 simulations
BASELINE_ELO_MCTS_MEDIUM = 1700       # MCTS with 128 simulations
BASELINE_ELO_MCTS_STRONG = 1900       # MCTS with 512 simulations

# Dec 28, 2025: Added 2000+ Elo baselines - required to measure models above 1900
BASELINE_ELO_MCTS_MASTER = 2000       # MCTS with 1024 simulations
BASELINE_ELO_MCTS_GRANDMASTER = 2100  # MCTS with 2048 simulations

# =============================================================================
# Evaluation Thresholds
# =============================================================================

# Shadow tournament interval (seconds)
SHADOW_TOURNAMENT_INTERVAL = 900  # 15 minutes

# Games per config in shadow tournaments
SHADOW_GAMES_PER_CONFIG = 15

# Full tournament interval (seconds)
FULL_TOURNAMENT_INTERVAL = 3600  # 1 hour

# Games in full tournaments
FULL_TOURNAMENT_GAMES = 50

# Minimum games for Elo calculation
MIN_GAMES_FOR_ELO = 30

# Elo K-factor for rating updates
ELO_K_FACTOR = 32

# =============================================================================
# Signal Weights (for training triggers)
# =============================================================================

# Weight for data freshness signal
SIGNAL_WEIGHT_FRESHNESS = 1.0

# Weight for model staleness signal
SIGNAL_WEIGHT_STALENESS = 0.8

# Weight for performance regression signal (higher = more urgent)
SIGNAL_WEIGHT_REGRESSION = 1.5

# Bootstrap priority multiplier
SIGNAL_BOOTSTRAP_PRIORITY = 10.0

# =============================================================================
# NNUE-Specific Thresholds
# =============================================================================

# Minimum games for NNUE value training
NNUE_MIN_GAMES = 10000

# Minimum games for NNUE policy training
NNUE_POLICY_MIN_GAMES = 5000

# Minimum games for CMA-ES heuristic optimization
CMAES_MIN_GAMES = 20000

# =============================================================================
# Plateau Detection
# =============================================================================

# Hours of no Elo improvement before plateau detected
PLATEAU_HOURS = 24

# Minimum Elo change to not be considered plateau
PLATEAU_MIN_ELO_CHANGE = 5

# Consecutive plateaus before architecture search
PLATEAU_TRIGGER_NAS = 3

# =============================================================================
# Resource Limits
# =============================================================================

# CPU utilization warning threshold
CPU_WARNING_PERCENT = 70

# CPU utilization critical threshold
CPU_CRITICAL_PERCENT = 80

# GPU utilization warning threshold
GPU_WARNING_PERCENT = 70

# GPU utilization critical threshold
GPU_CRITICAL_PERCENT = 80

# Memory utilization warning threshold
MEMORY_WARNING_PERCENT = 70

# Memory utilization critical threshold
MEMORY_CRITICAL_PERCENT = 80

# Disk utilization warning threshold
DISK_WARNING_PERCENT = 65

# Disk utilization critical threshold
DISK_CRITICAL_PERCENT = 70

# =============================================================================
# File Descriptor & Socket Limits (December 2025)
# =============================================================================
# Prevents "too many open files" crashes in P2P and long-running processes

# File descriptor utilization warning threshold (percent of limit)
FD_WARNING_PERCENT = 80

# File descriptor utilization critical threshold (percent of limit)
FD_CRITICAL_PERCENT = 90

# Socket connection warning thresholds
SOCKET_TIME_WAIT_WARNING = 100  # Many TIME_WAIT indicates connection churn
SOCKET_TIME_WAIT_CRITICAL = 500  # Likely connection leak

SOCKET_CLOSE_WAIT_WARNING = 20  # CLOSE_WAIT means remote closed, we haven't
SOCKET_CLOSE_WAIT_CRITICAL = 50  # Definite resource leak

# Total socket connection thresholds (per process)
SOCKET_TOTAL_WARNING = 200
SOCKET_TOTAL_CRITICAL = 500

# =============================================================================
# Network/SSH Timeouts
# =============================================================================

# SSH connection timeout (seconds)
SSH_CONNECT_TIMEOUT = 10

# SSH command timeout (seconds)
SSH_COMMAND_TIMEOUT = 30

# HTTP request timeout (seconds)
HTTP_TIMEOUT = 30

# P2P operations timeout (seconds)
P2P_TIMEOUT = 30

# =============================================================================
# Training Pipeline Timeouts (December 2025)
# =============================================================================

# SQLite connection timeout (seconds)
SQLITE_TIMEOUT = 30

# SQLite short operations timeout (seconds)
SQLITE_SHORT_TIMEOUT = 10

# SQLite PRAGMA settings (December 2025 - consolidated)
# busy_timeout in milliseconds - how long to wait for locks
# December 29, 2025: Increased standard from 10s to 30s for cluster operations
# This prevents lock timeout errors during concurrent selfplay/training
SQLITE_BUSY_TIMEOUT_MS = 30000  # 30 seconds (standard for cluster ops)
SQLITE_BUSY_TIMEOUT_LONG_MS = 60000  # 60 seconds (for heavy merge operations)
SQLITE_BUSY_TIMEOUT_SHORT_MS = 5000  # 5 seconds (for quick registry ops)

# Journal mode - WAL for better concurrency
SQLITE_JOURNAL_MODE = "WAL"

# Synchronous mode - NORMAL for performance with reasonable safety
SQLITE_SYNCHRONOUS = "NORMAL"

# WAL autocheckpoint - pages before automatic checkpoint
SQLITE_WAL_AUTOCHECKPOINT = 100

# Cache size in KB (negative = KB, positive = pages)
SQLITE_CACHE_SIZE_KB = -2000  # 2MB cache

# URL open timeout for quick health checks (seconds)
URLOPEN_SHORT_TIMEOUT = 5

# URL open timeout for data operations (seconds)
URLOPEN_TIMEOUT = 10

# Rsync transfer timeout (seconds) - increased for large DB transfers
RSYNC_TIMEOUT = 300

# Async subprocess wait timeout (seconds)
ASYNC_SUBPROCESS_TIMEOUT = 180

# Process/thread join timeout (seconds)
THREAD_JOIN_TIMEOUT = 5

# Future result timeout for parallel operations (seconds)
FUTURE_RESULT_TIMEOUT = 300

# Checkpoint future timeout (seconds)
CHECKPOINT_FUTURE_TIMEOUT = 120

# Long training job timeout (seconds) - 4 hours
TRAINING_JOB_TIMEOUT = 14400

# Training lock timeout (seconds) - 2 hours
TRAINING_LOCK_TIMEOUT = 7200

# Resource wait timeout (seconds)
RESOURCE_WAIT_TIMEOUT = 300

# =============================================================================
# Cluster Health & Monitoring
# =============================================================================

# Heartbeat interval for health checks (seconds)
HEARTBEAT_INTERVAL = 30

# Peer timeout - no heartbeat means dead (seconds)
PEER_TIMEOUT = 90

# Election timeout for leader election (seconds)
ELECTION_TIMEOUT = 10

# Leader lease renewal interval (seconds)
LEADER_LEASE_RENEW_INTERVAL = 10

# Job status check interval (seconds)
# Dec 29, 2025: Reduced from 60s to 15s for faster job status updates
JOB_CHECK_INTERVAL = 15

# Peer discovery broadcast interval (seconds)
DISCOVERY_INTERVAL = 120

# Stale entry cleanup interval (seconds)
STALE_CLEANUP_INTERVAL = 60

# =============================================================================
# Data Sync Intervals
# =============================================================================

# Main sync cycle interval (seconds)
SYNC_INTERVAL = 60

# Transport health check interval (seconds)
TRANSPORT_HEALTH_CHECK_INTERVAL = 300  # 5 minutes

# Freshness check interval (seconds)
FRESHNESS_CHECK_INTERVAL = 60

# Checkpoint interval for ephemeral data (seconds)
CHECKPOINT_INTERVAL = 60

# Stale data thresholds (December 2025)
STALE_DATA_THRESHOLD_SECONDS = 1800  # 30 minutes - data older than this is stale
CRITICAL_STALE_THRESHOLD_SECONDS = 3600  # 1 hour - urgent sync needed

# Max items in sync queue
MAX_SYNC_QUEUE_SIZE = 20

# =============================================================================
# Lock & Mutex Settings
# =============================================================================

# Default lock timeout (seconds)
DEFAULT_LOCK_TIMEOUT = 3600  # 1 hour

# Lock acquisition timeout (seconds)
DEFAULT_ACQUIRE_TIMEOUT = 60

# Lock poll interval when waiting (seconds)
LOCK_POLL_INTERVAL = 0.5

# Crash detection threshold - no heartbeat (seconds)
CRASH_DETECTION_THRESHOLD = 60

# =============================================================================
# Cluster Transport Timeouts (December 2025)
# =============================================================================

# TCP connection timeout (seconds) - longer for VAST.ai
CLUSTER_CONNECT_TIMEOUT = 30

# Operation timeout for cluster commands (seconds)
CLUSTER_OPERATION_TIMEOUT = 180

# P2P HTTP request timeout (seconds)
P2P_HTTP_TIMEOUT = 30

# =============================================================================
# Circuit Breaker & Retry
# =============================================================================

# Number of failures before circuit opens
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3

# Time to wait before attempting recovery (seconds)
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 300  # 5 minutes

# Per-operation circuit breaker configurations (December 2025)
# Keys: failure_threshold (int), recovery_timeout (float in seconds)
CIRCUIT_BREAKER_CONFIGS = {
    "ssh": {"failure_threshold": 3, "recovery_timeout": 60.0},
    "http": {"failure_threshold": 5, "recovery_timeout": 30.0},
    "p2p": {"failure_threshold": 3, "recovery_timeout": 45.0},
    "aria2": {"failure_threshold": 2, "recovery_timeout": 120.0},
    "rsync": {"failure_threshold": 2, "recovery_timeout": 90.0},
    "database": {"failure_threshold": 3, "recovery_timeout": 30.0},
    "external_api": {"failure_threshold": 5, "recovery_timeout": 60.0},
}

# Maximum retry attempts for recoverable errors
MAX_RETRY_ATTEMPTS = 3

# Base delay between retries (seconds)
RETRY_BASE_DELAY = 1.0

# Maximum delay between retries (seconds)
RETRY_MAX_DELAY = 60.0

# Training retry sleep (seconds)
# Dec 29, 2025: Reduced from 10s to 2s for faster failure recovery
# Configurable via RINGRIFT_TRAINING_RETRY_SLEEP env var
import os as _os
TRAINING_RETRY_SLEEP_SECONDS = float(_os.getenv("RINGRIFT_TRAINING_RETRY_SLEEP", "2.0"))

# =============================================================================
# Model Management
# =============================================================================

# Model count threshold to trigger culling
MODEL_CULL_THRESHOLD = 100

# Win rate threshold for model promotion
# Dec 2025: Raised from 0.60 to 0.80 to reach 2000+ Elo.
# 60% only reaches ~1400 Elo plateau. 80% vs heuristic = ~1600+ Elo baseline.
PROMOTION_WIN_RATE_THRESHOLD = 0.80  # 80% vs heuristic for strong model promotion

# ELO underserved threshold (fewer games = underserved)
ELO_UNDERSERVED_THRESHOLD = 100

# =============================================================================
# File Transfer
# =============================================================================

# Maximum timeout for batch operations (seconds)
MAX_BATCH_TIMEOUT = 1800  # 30 minutes

# Maximum timeout per file (seconds)
MAX_PER_FILE_TIMEOUT = 120  # 2 minutes

# Rsync operation timeouts (December 2025 - consolidated)
RSYNC_TIMEOUT = 300  # Per-file I/O timeout (seconds) - increased for large DBs
RSYNC_BATCH_TIMEOUT = 600  # Batch rsync timeout (seconds)
RSYNC_MAX_TIMEOUT = 1200  # Maximum transfer time (seconds)

# Ephemeral sync configuration
EPHEMERAL_SYNC_TIMEOUT = 30  # Quick sync timeout for volatile hosts
EPHEMERAL_SYNC_INTERVAL = 5  # Aggressive polling interval (seconds)

# =============================================================================
# Shadow Validation
# =============================================================================

# Maximum divergence allowed between implementations
DIVERGENCE_THRESHOLD = 0.001  # 0.1%


# =============================================================================
# Distillation & Temperature (December 2025)
# =============================================================================

# Knowledge distillation temperature (higher = softer)
DISTILL_TEMPERATURE = 3.0

# Distillation loss weight (alpha between student/teacher)
DISTILL_ALPHA = 0.7

# Value head distillation temperature
VALUE_TEMPERATURE = 1.0

# Policy head distillation temperature
POLICY_TEMPERATURE = 3.0

# EMA decay for model averaging
EMA_DECAY = 0.999

# Focal loss gamma for hard example mining
FOCAL_GAMMA = 2.0

# Policy label smoothing factor
POLICY_LABEL_SMOOTHING = 0.05


# =============================================================================
# Quality Thresholds (December 2025)
# =============================================================================

# Minimum quality score for training data (absolute floor)
MIN_QUALITY_FOR_TRAINING = 0.3

# Low quality threshold - triggers warnings and exploration boost
LOW_QUALITY_THRESHOLD = 0.4

# Minimum quality score for priority sync (higher than training minimum)
MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5

# Minimum games before sync is worthwhile (Phase 3 Dec 2025)
# Below this threshold, sync overhead exceeds benefit
MIN_GAMES_FOR_SYNC = 50

# Medium quality threshold - baseline for "acceptable" quality
MEDIUM_QUALITY_THRESHOLD = 0.6

# High quality threshold (affects priority)
HIGH_QUALITY_THRESHOLD = 0.7

# Good quality threshold - accelerated training path
QUALITY_GOOD_THRESHOLD = 0.80

# Excellent quality threshold - hot path for fast promotion
QUALITY_EXCELLENT_THRESHOLD = 0.90

# Overfit detection threshold (train-val gap)
OVERFIT_THRESHOLD = 0.15

# =============================================================================
# Gauntlet Feedback Thresholds (December 2025)
# =============================================================================

# Strong performance vs heuristic - reduces exploration (model is performing well)
STRONG_VS_HEURISTIC_THRESHOLD = 0.80

# Weak performance vs random - indicates need for more training
WEAK_VS_RANDOM_THRESHOLD = 0.70


# =============================================================================
# Selfplay & Gameplay (December 2025)
# =============================================================================

# Maximum moves per game before termination
SELFPLAY_MAX_MOVES = 10000

# Base temperature for move selection
SELFPLAY_TEMPERATURE = 1.0

# Higher temperature for opening moves (diversity)
SELFPLAY_OPENING_TEMPERATURE = 1.5

# Quality threshold for adaptive engine selection
SELFPLAY_QUALITY_THRESHOLD = 0.7


# =============================================================================
# GPU Batch Scaling (December 2025)
# =============================================================================

# Reserved GPU memory for system overhead (GB)
GPU_RESERVED_MEMORY_GB = 8.0

# Maximum batch size regardless of memory
GPU_MAX_BATCH_SIZE = 16384

# Batch multipliers by GPU type (base batch = 64)
GH200_BATCH_MULTIPLIER = 64  # NVIDIA GH200 (96GB)
H100_BATCH_MULTIPLIER = 32   # NVIDIA H100 (80GB)
A100_BATCH_MULTIPLIER = 16   # NVIDIA A100 (40/80GB)
A10_BATCH_MULTIPLIER = 8     # NVIDIA A10 (24GB)
RTX_BATCH_MULTIPLIER = 4     # RTX 4090/3090 (24GB)


# =============================================================================
# Coordination & Concurrency (December 2025)
# =============================================================================

# Idle GPU utilization threshold (below = available)
IDLE_GPU_THRESHOLD = 10

# Default P2P communication port
# NOTE: Canonical location is app/config/ports.py - import from there for new code
# Kept for backward compatibility; may be removed in future versions
from app.config.ports import P2P_DEFAULT_PORT  # noqa: F401

# Ephemeral data evacuation threshold (games)
EVACUATION_THRESHOLD = 50

# Maximum concurrent syncs per host
MAX_CONCURRENT_SYNCS_PER_HOST = 1

# Maximum global concurrent syncs
MAX_CONCURRENT_SYNCS_GLOBAL = 5

# Target GPU utilization range
TARGET_GPU_UTILIZATION_MIN = 60
TARGET_GPU_UTILIZATION_MAX = 80


# =============================================================================
# Data Streaming (December 2025)
# =============================================================================

# Streaming buffer size (games)
DATA_STREAMING_BUFFER_SIZE = 10000

# Minimum buffer fill before training starts (0-1)
DATA_STREAMING_MIN_BUFFER_FILL = 0.2

# Deduplication window size (games)
DATA_STREAMING_DEDUPE_WINDOW = 50000

# Weighting factors for data sampling
DATA_SAMPLING_RECENCY_WEIGHT = 0.3
DATA_SAMPLING_QUALITY_WEIGHT = 0.4


# =============================================================================
# Board-Type Specific Promotion Thresholds (December 2025)
# =============================================================================
# Two-tier promotion system:
# 1. ASPIRATIONAL thresholds - target for 2000+ Elo models
# 2. MINIMUM thresholds - absolute floor to avoid promoting garbage
# 3. RELATIVE promotion - if model beats current best, promote even if below aspirational
#
# This prevents progress from stalling when no models reach aspirational targets yet.

# Aspirational thresholds - these are the targets for strong models
PROMOTION_THRESHOLDS_BY_CONFIG: dict[str, dict[str, float]] = {
    # Hex8 (61 cells) - smaller, easier
    "hex8_2p": {"vs_random": 0.90, "vs_heuristic": 0.85},
    "hex8_3p": {"vs_random": 0.65, "vs_heuristic": 0.55},
    "hex8_4p": {"vs_random": 0.60, "vs_heuristic": 0.50},

    # Square8 (64 cells) - small, moderate difficulty
    "square8_2p": {"vs_random": 0.90, "vs_heuristic": 0.85},
    "square8_3p": {"vs_random": 0.65, "vs_heuristic": 0.55},
    "square8_4p": {"vs_random": 0.60, "vs_heuristic": 0.50},

    # Square19 (361 cells) - large, harder
    "square19_2p": {"vs_random": 0.85, "vs_heuristic": 0.70},
    "square19_3p": {"vs_random": 0.60, "vs_heuristic": 0.45},
    "square19_4p": {"vs_random": 0.55, "vs_heuristic": 0.40},

    # Hexagonal (469 cells) - largest, hardest
    "hexagonal_2p": {"vs_random": 0.85, "vs_heuristic": 0.65},
    "hexagonal_3p": {"vs_random": 0.55, "vs_heuristic": 0.40},
    "hexagonal_4p": {"vs_random": 0.50, "vs_heuristic": 0.35},
}

# Dec 28, 2025: Minimum floor thresholds for relative promotion
# If model beats current best AND meets minimum floor, allow promotion
# This prevents progress from stalling while still avoiding garbage models
PROMOTION_MINIMUM_THRESHOLDS: dict[str, dict[str, float]] = {
    # 2-player: Must beat random decisively (70%+), any heuristic performance OK
    "hex8_2p": {"vs_random": 0.70, "vs_heuristic": 0.0},
    "square8_2p": {"vs_random": 0.70, "vs_heuristic": 0.0},
    "square19_2p": {"vs_random": 0.65, "vs_heuristic": 0.0},
    "hexagonal_2p": {"vs_random": 0.65, "vs_heuristic": 0.0},

    # 3-player: Lower bar due to increased complexity
    "hex8_3p": {"vs_random": 0.50, "vs_heuristic": 0.0},
    "square8_3p": {"vs_random": 0.50, "vs_heuristic": 0.0},
    "square19_3p": {"vs_random": 0.45, "vs_heuristic": 0.0},
    "hexagonal_3p": {"vs_random": 0.45, "vs_heuristic": 0.0},

    # 4-player: Even lower bar
    "hex8_4p": {"vs_random": 0.40, "vs_heuristic": 0.0},
    "square8_4p": {"vs_random": 0.40, "vs_heuristic": 0.0},
    "square19_4p": {"vs_random": 0.35, "vs_heuristic": 0.0},
    "hexagonal_4p": {"vs_random": 0.35, "vs_heuristic": 0.0},
}

# Enable relative promotion (beat-the-champion mode)
# When True: Promote if (meets minimum floor) AND (beats current best model)
# When False: Only promote if meets aspirational thresholds
PROMOTION_RELATIVE_ENABLED: bool = True


def get_promotion_thresholds(config_key: str) -> dict[str, float]:
    """Get ASPIRATIONAL promotion thresholds for a configuration.

    These are the target thresholds for strong (2000+ Elo) models.
    For relative promotion, use should_promote_model() instead.

    Args:
        config_key: Configuration key like 'hex8_2p', 'square19_4p'

    Returns:
        Dict with 'vs_random' and 'vs_heuristic' thresholds
    """
    if config_key in PROMOTION_THRESHOLDS_BY_CONFIG:
        return PROMOTION_THRESHOLDS_BY_CONFIG[config_key]

    # Parse config to get player count for fallback
    if "_4p" in config_key:
        return {"vs_random": 0.60, "vs_heuristic": 0.50}
    elif "_3p" in config_key:
        return {"vs_random": 0.65, "vs_heuristic": 0.55}
    else:
        return {"vs_random": 0.90, "vs_heuristic": 0.85}


def get_minimum_thresholds(config_key: str) -> dict[str, float]:
    """Get MINIMUM floor thresholds for relative promotion.

    These are the absolute minimum requirements - models below this
    are garbage and should never be promoted.

    Args:
        config_key: Configuration key like 'hex8_2p', 'square19_4p'

    Returns:
        Dict with 'vs_random' and 'vs_heuristic' minimum thresholds
    """
    if config_key in PROMOTION_MINIMUM_THRESHOLDS:
        return PROMOTION_MINIMUM_THRESHOLDS[config_key]

    # Parse config to get player count for fallback
    if "_4p" in config_key:
        return {"vs_random": 0.35, "vs_heuristic": 0.0}
    elif "_3p" in config_key:
        return {"vs_random": 0.45, "vs_heuristic": 0.0}
    else:
        return {"vs_random": 0.65, "vs_heuristic": 0.0}


def should_promote_model(
    config_key: str,
    vs_random_rate: float,
    vs_heuristic_rate: float,
    beats_current_best: bool = False,
) -> tuple[bool, str]:
    """Determine if a model should be promoted using two-tier system.

    Promotion criteria (in order of precedence):
    1. If model meets ASPIRATIONAL thresholds -> promote (strong model)
    2. If PROMOTION_RELATIVE_ENABLED and beats_current_best:
       - If model meets MINIMUM floor -> promote (incremental improvement)
    3. Otherwise -> don't promote

    Args:
        config_key: Configuration key like 'hex8_2p'
        vs_random_rate: Win rate against random opponent (0.0-1.0)
        vs_heuristic_rate: Win rate against heuristic opponent (0.0-1.0)
        beats_current_best: Whether this model beats the current best model

    Returns:
        Tuple of (should_promote, reason)
    """
    aspirational = get_promotion_thresholds(config_key)
    minimum = get_minimum_thresholds(config_key)

    # Check aspirational thresholds first
    if vs_random_rate >= aspirational["vs_random"] and vs_heuristic_rate >= aspirational["vs_heuristic"]:
        return True, f"Meets aspirational targets (vs_random={vs_random_rate:.1%} >= {aspirational['vs_random']:.0%}, vs_heuristic={vs_heuristic_rate:.1%} >= {aspirational['vs_heuristic']:.0%})"

    # Check relative promotion
    if PROMOTION_RELATIVE_ENABLED and beats_current_best:
        if vs_random_rate >= minimum["vs_random"]:
            return True, f"Beats current best and meets minimum floor (vs_random={vs_random_rate:.1%} >= {minimum['vs_random']:.0%})"
        else:
            return False, f"Beats current best but below minimum floor (vs_random={vs_random_rate:.1%} < {minimum['vs_random']:.0%})"

    # Didn't meet any criteria
    if vs_random_rate < aspirational["vs_random"]:
        return False, f"Below aspirational vs_random ({vs_random_rate:.1%} < {aspirational['vs_random']:.0%})"
    else:
        return False, f"Below aspirational vs_heuristic ({vs_heuristic_rate:.1%} < {aspirational['vs_heuristic']:.0%})"


# =============================================================================
# Resource Allocation Constants (December 2025)
# =============================================================================
# Node capability weighting for adaptive workload distribution.

# GPU memory weights (relative to A100 40GB baseline)
GPU_MEMORY_WEIGHTS: dict[str, float] = {
    "GH200": 2.4,    # 96GB - 2.4x weight
    "H200": 2.0,     # 80GB - 2x weight
    "H100": 2.0,     # 80GB - 2x weight
    "A100_80": 2.0,  # 80GB - 2x weight
    "A100": 1.0,     # 40GB - baseline
    "A10": 0.6,      # 24GB - 0.6x weight
    "RTX_4090": 0.6, # 24GB - 0.6x weight
    "RTX_3090": 0.6, # 24GB - 0.6x weight
    "CPU": 0.1,      # CPU-only - minimal weight
}

# Selfplay games allocation per node type (per batch)
SELFPLAY_GAMES_PER_NODE: dict[str, int] = {
    "GH200": 2000,   # High memory, fast
    "H200": 1500,
    "H100": 1500,
    "A100_80": 1200,
    "A100": 800,
    "A10": 400,
    "RTX_4090": 400,
    "RTX_3090": 300,
    "CPU": 50,       # CPU selfplay is slow
}
# Alias for GPU-centric code
SELFPLAY_GAMES_PER_GPU_TYPE = SELFPLAY_GAMES_PER_NODE

# Training batch size by GPU type
TRAINING_BATCH_SIZE_BY_GPU: dict[str, int] = {
    "GH200": 2048,
    "H200": 1536,
    "H100": 1536,
    "A100_80": 1024,
    "A100": 512,
    "A10": 256,
    "RTX_4090": 256,
    "RTX_3090": 256,
    "CPU": 64,
}

# Maximum concurrent gauntlets per node type
MAX_CONCURRENT_GAUNTLETS_BY_GPU: dict[str, int] = {
    "GH200": 4,
    "H200": 3,
    "H100": 3,
    "A100_80": 3,
    "A100": 2,
    "A10": 1,
    "RTX_4090": 1,
    "RTX_3090": 1,
    "CPU": 1,
}

# Node roles for workload distribution
NODE_ROLES = ["training", "selfplay", "coordinator", "ephemeral", "cpu_only"]

# Ephemeral node patterns (Vast.ai, RunPod, AWS Spot)
EPHEMERAL_NODE_PATTERNS = ["vast-", "runpod-", "spot-", "ephemeral-"]

# Minimum resources for task scheduling
MIN_MEMORY_GB_FOR_TRAINING = 32
MIN_MEMORY_GB_FOR_SELFPLAY = 16
MIN_MEMORY_GB_FOR_GAUNTLET = 16


def get_gpu_weight(gpu_type: str) -> float:
    """Get workload weight multiplier for a GPU type."""
    return GPU_MEMORY_WEIGHTS.get(gpu_type, 1.0)


def get_selfplay_allocation(gpu_type: str) -> int:
    """Get recommended selfplay games per batch for a GPU type."""
    return SELFPLAY_GAMES_PER_NODE.get(gpu_type, 500)


def is_ephemeral_node(node_id: str) -> bool:
    """Check if a node is ephemeral (temporary cloud instance)."""
    return any(pattern in node_id.lower() for pattern in EPHEMERAL_NODE_PATTERNS)


# =============================================================================
# Quality Gating Thresholds (December 2025)
# =============================================================================
# Thresholds for blocking training on low-quality data.

# Minimum samples required to trigger training
MIN_SAMPLES_FOR_TRAINING = 10_000

# Minimum average game length (moves) for quality games
MIN_AVG_GAME_LENGTH = 20

# Maximum draw rate - high draw rate indicates weak play
MAX_DRAW_RATE_FOR_TRAINING = 0.25

# Minimum win rate vs heuristic in recent selfplay
MIN_SELFPLAY_WIN_RATE_VS_HEURISTIC = 0.40

# Data staleness threshold (hours) - data older than this triggers warning
DATA_STALENESS_WARNING_HOURS = 4.0

# Critical staleness - data older than this blocks training
DATA_STALENESS_CRITICAL_HOURS = 24.0


def check_training_data_quality(
    sample_count: int,
    avg_game_length: float,
    draw_rate: float,
    win_rate_vs_heuristic: float,
) -> tuple[bool, list[str]]:
    """Check if training data meets quality thresholds.

    Args:
        sample_count: Number of training samples
        avg_game_length: Average game length in moves
        draw_rate: Fraction of games ending in draw
        win_rate_vs_heuristic: Recent win rate vs heuristic

    Returns:
        Tuple of (passes_quality, list_of_issues)
    """
    issues = []

    if sample_count < MIN_SAMPLES_FOR_TRAINING:
        issues.append(f"Insufficient samples: {sample_count} < {MIN_SAMPLES_FOR_TRAINING}")

    if avg_game_length < MIN_AVG_GAME_LENGTH:
        issues.append(f"Games too short: avg {avg_game_length:.1f} < {MIN_AVG_GAME_LENGTH}")

    if draw_rate > MAX_DRAW_RATE_FOR_TRAINING:
        issues.append(f"Draw rate too high: {draw_rate:.1%} > {MAX_DRAW_RATE_FOR_TRAINING:.1%}")

    if win_rate_vs_heuristic < MIN_SELFPLAY_WIN_RATE_VS_HEURISTIC:
        issues.append(
            f"Selfplay too weak: {win_rate_vs_heuristic:.1%} vs heuristic "
            f"< {MIN_SELFPLAY_WIN_RATE_VS_HEURISTIC:.1%}"
        )

    return len(issues) == 0, issues


# =============================================================================
# Feedback Loop Thresholds (December 2025)
# =============================================================================

# ELO momentum lookback (number of updates)
ELO_MOMENTUM_LOOKBACK = 5

# Strong ELO improvement threshold (per update)
ELO_STRONG_IMPROVEMENT = 25.0

# Moderate ELO improvement threshold
ELO_MODERATE_IMPROVEMENT = 12.0

# ELO plateau threshold (below = stagnant)
ELO_PLATEAU_THRESHOLD = 5.0

# Training intensity multiplier range
MAX_INTENSITY_MULTIPLIER = 2.5
MIN_INTENSITY_MULTIPLIER = 0.5

# -----------------------------------------------------------------------------
# FeedbackLoopController Constants (December 28, 2025)
# Extracted from feedback_loop_controller.py to enable configuration
# -----------------------------------------------------------------------------

# Policy accuracy threshold - trigger evaluation above this
POLICY_ACCURACY_EVALUATION_THRESHOLD = 0.75

# Exploration boost on promotion failure
FAILURE_EXPLORATION_BOOST = 1.3

# Intensity reduction on promotion success
SUCCESS_INTENSITY_REDUCTION = 0.9

# ELO change that triggers curriculum rebalancing
ELO_SIGNIFICANT_CHANGE = 30.0

# Cooldown after plateau detection (seconds)
PLATEAU_COOLDOWN_SECONDS = 600.0

# Cooldown between tournament triggers (seconds)
TOURNAMENT_COOLDOWN_SECONDS = 300.0

# Exploration boost multiplier on failure (applied repeatedly, up to max)
EXPLORATION_BOOST_MULTIPLIER = 1.2

# Maximum exploration boost value
EXPLORATION_BOOST_MAX = 2.0

# Exploration boost for recovery scenarios
EXPLORATION_BOOST_RECOVERY = 1.5

# Rate change threshold - significant selfplay rate change (percent)
RATE_CHANGE_SIGNIFICANT_PERCENT = 20.0

# Curriculum weight adjustments
CURRICULUM_WEIGHT_ADJUSTMENT_DOWN = -0.05  # Reduce weight (less urgent)
CURRICULUM_WEIGHT_ADJUSTMENT_UP = 0.10     # Increase weight (more urgent)

# Policy accuracy thresholds for weight adjustment
POLICY_LOW_THRESHOLD = 0.65   # Below this: boost training weight
POLICY_HIGH_THRESHOLD = 0.80  # Above this: reduce training weight

# ELO velocity thresholds (per hour)
ELO_PLATEAU_PER_HOUR = 10.0          # Below this is considered plateau
ELO_FAST_IMPROVEMENT_PER_HOUR = 50.0  # Above this is fast improvement

# Loss anomaly severity threshold (consecutive anomalies)
LOSS_ANOMALY_SEVERE_COUNT = 3

# Exploration boost per anomaly (up to max)
EXPLORATION_BOOST_PER_ANOMALY = 0.15

# Exploration boost per stall epochs (stall_epochs // 5)
EXPLORATION_BOOST_PER_STALL_GROUP = 0.10
EXPLORATION_BOOST_STALL_MAX = 1.5

# Exploration boost decay factor (reduce toward 1.0)
EXPLORATION_BOOST_DECAY = 0.9

# Exploration boost adjustments on promotion result
EXPLORATION_BOOST_BASE = 1.0              # Minimum (floor) value
EXPLORATION_BOOST_SUCCESS_DECREMENT = 0.1  # Decrease on success
EXPLORATION_BOOST_FAILURE_INCREMENT = 0.2  # Increase on failure

# Trend duration thresholds (epochs)
TREND_DURATION_MODERATE = 3   # Moderate intervention
TREND_DURATION_SEVERE = 5     # Severe intervention

# Consecutive success threshold before reducing exploration
CONSECUTIVE_SUCCESS_THRESHOLD = 3


# =============================================================================
# Alert Levels and Monitoring Thresholds
# =============================================================================
# Consolidated from app/monitoring/thresholds.py

from enum import Enum
from typing import Any, Optional


class AlertLevel(str, Enum):
    """Alert severity levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


# Master threshold configuration for monitoring systems
# All monitoring scripts should reference these values instead of hardcoding
MONITORING_THRESHOLDS: dict[str, dict[str, Any]] = {
    # Disk space monitoring
    "disk": {
        "warning": DISK_WARNING_PERCENT,
        "critical": DISK_CRITICAL_PERCENT,
        "fatal": 95,
        "unit": "percent",
        "description": "Disk space utilization thresholds",
    },

    # GPU monitoring
    "gpu_utilization": {
        "idle": 5,
        "low": 20,
        "normal": 50,
        "warning": GPU_WARNING_PERCENT,
        "critical": GPU_CRITICAL_PERCENT,
        "unit": "percent",
        "description": "GPU utilization levels",
    },
    "gpu_memory": {
        "warning": 85,
        "critical": 95,
        "unit": "percent",
        "description": "GPU memory utilization",
    },

    # Training monitoring
    "training": {
        "stale_hours": TRAINING_STALENESS_HOURS,
        "model_stale_hours": 48,
        "min_batch_rate": 10,
        "max_loss_increase": 0.5,
        "description": "Training progress thresholds",
    },

    # Data quality monitoring
    "data_quality": {
        "draw_rate_threshold": 0.20,
        "min_game_length": 10,
        "max_game_length": 500,
        "nan_threshold": 0.001,
        "zero_feature_threshold": 0.05,
        "description": "Data quality metrics",
    },

    # Cluster health
    "cluster": {
        "min_nodes_online": 5,
        "node_timeout_seconds": PEER_TIMEOUT,
        "heartbeat_interval": HEARTBEAT_INTERVAL,
        "max_coordinator_lag_seconds": 300,
        "description": "Cluster health requirements",
    },

    # Selfplay monitoring
    "selfplay": {
        "min_games_per_hour": 100,
        "max_game_duration_seconds": 600,
        "min_move_count": 5,
        "description": "Selfplay generation metrics",
    },

    # Network/P2P monitoring
    "network": {
        "ping_timeout_ms": 5000,
        "max_relay_latency_ms": 200,
        "reconnect_interval_seconds": 30,
        "description": "Network health thresholds",
    },

    # Memory monitoring
    "memory": {
        "warning": MEMORY_WARNING_PERCENT,
        "critical": MEMORY_CRITICAL_PERCENT,
        "unit": "percent",
        "description": "System memory thresholds",
    },
}

# Backwards compatibility alias
THRESHOLDS = MONITORING_THRESHOLDS


def get_threshold(
    category: str,
    key: str,
    default: Optional[Any] = None,
) -> Any:
    """Get a specific monitoring threshold value.

    Args:
        category: Threshold category (e.g., "disk", "gpu_utilization")
        key: Specific threshold key (e.g., "warning", "critical")
        default: Default value if not found

    Returns:
        Threshold value or default

    Example:
        disk_warning = get_threshold("disk", "warning")  # Returns 65
    """
    if category not in MONITORING_THRESHOLDS:
        return default
    return MONITORING_THRESHOLDS[category].get(key, default)


def should_alert(
    category: str,
    value: float,
    level: str = "warning",
    comparison: str = "gte",
) -> bool:
    """Check if a value exceeds the threshold for alert.

    Args:
        category: Threshold category
        value: Current value to check
        level: Alert level to check against
        comparison: Comparison type (gte=>=, lte=<=, gt=>, lt=<, eq===)

    Returns:
        True if value triggers alert at specified level

    Example:
        if should_alert("disk", 75, "warning"):
            send_warning()
    """
    threshold = get_threshold(category, level)
    if threshold is None:
        return False

    comparisons = {
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "eq": lambda v, t: v == t,
    }

    compare_fn = comparisons.get(comparison, comparisons["gte"])
    return compare_fn(value, threshold)


def get_all_thresholds() -> dict[str, dict[str, Any]]:
    """Get all monitoring thresholds for display/documentation."""
    return MONITORING_THRESHOLDS.copy()


def update_threshold(category: str, key: str, value: Any) -> None:
    """Update a threshold value at runtime.

    Use sparingly - primarily for testing or dynamic configuration.
    """
    if category in MONITORING_THRESHOLDS:
        MONITORING_THRESHOLDS[category][key] = value


# =============================================================================
# Training Control Thresholds (December 2025)
# =============================================================================

# Checkpoint timeout for async save operations (seconds)
CHECKPOINT_TIMEOUT = 120

# December 2025: Increased patience values to allow more training.
# Previous values (10) caused training to stop after 3-5 epochs,
# but 20-30 epochs are needed to reach 2000+ Elo.
#
# Dec 28, 2025: Further increased - 15 epochs still plateaus at 1650 Elo.
# Models need 20+ epochs to learn deep tactics for 1800+ Elo.
#
# Minimum training epochs before early stopping can trigger
MIN_TRAINING_EPOCHS = 20

# Validation patience - epochs without improvement before early stopping
VALIDATION_PATIENCE = 22

# Elo plateau patience - updates without Elo improvement before early stopping
ELO_PATIENCE = 20

# Learning rate warmup steps (linear warmup from 0 to lr)
LR_WARMUP_STEPS = 0  # Default: no warmup (set via CLI --warmup-steps)

# Early stopping patience - epochs without validation improvement
EARLY_STOPPING_PATIENCE = 20

# December 29, 2025: Phase 9 - Board-specific base patience values
# Weak models on hard boards stop too early with uniform patience
EARLY_STOPPING_PATIENCE_BY_BOARD = {
    "hex8": 5,       # Smallest board, learns fastest
    "square8": 5,    # Similar to hex8
    "square19": 10,  # Large board needs more epochs
    "hexagonal": 12, # Largest board, slowest to learn
}


def _round_elo_for_cache(elo: float) -> int:
    """Round Elo to nearest 100 for efficient caching.

    Since patience changes at 700 and 1000 Elo thresholds,
    rounding to 100 provides good granularity without
    cache misses on small Elo fluctuations.
    """
    return int(round(elo / 100) * 100)


@lru_cache(maxsize=128)
def _get_adaptive_patience_cached(board: str, players: int, elo_bucket: int) -> int:
    """Cached implementation of adaptive patience calculation.

    December 29, 2025: Internal cached version.
    Uses elo_bucket (rounded Elo) as cache key.

    Args:
        board: Board type (hex8, square8, square19, hexagonal)
        players: Number of players (2, 3, or 4)
        elo_bucket: Elo rounded to nearest 100

    Returns:
        Patience value (epochs without improvement before early stopping)
    """
    # Base patience by board type
    patience = EARLY_STOPPING_PATIENCE_BY_BOARD.get(board, 5)

    # +2 epochs per additional player (multiplayer more complex)
    patience += (players - 2) * 2

    # Adjust for model strength
    if elo_bucket < 700:
        # Very weak models need more patience (50% more)
        patience = int(patience * 1.5)
    elif elo_bucket < 1000:
        # Weak models need some extra patience (20% more)
        patience = int(patience * 1.2)
    # Strong models (>1000 Elo) use base patience

    # Ensure minimum patience
    return max(patience, 3)


def get_adaptive_patience(board: str, players: int, elo: float) -> int:
    """Get early stopping patience adapted to config difficulty.

    December 29, 2025: Phase 9 - Adaptive early stopping patience.
    Weak models on hard boards stop too early with uniform patience.
    This function computes patience based on:
    - Board type (larger boards need more patience)
    - Player count (multiplayer needs more patience)
    - Current Elo (weak models need more patience)

    Note: Results are cached with Elo rounded to nearest 100
    for efficiency. Small Elo fluctuations won't cause cache misses.

    Args:
        board: Board type (hex8, square8, square19, hexagonal)
        players: Number of players (2, 3, or 4)
        elo: Current Elo rating for the config

    Returns:
        Patience value (epochs without improvement before early stopping)
    """
    elo_bucket = _round_elo_for_cache(elo)
    return _get_adaptive_patience_cached(board, players, elo_bucket)


# Validation interval - steps between validation checks
VALIDATION_INTERVAL_STEPS = 500

# Checkpoint save interval - epochs between checkpoint saves
CHECKPOINT_SAVE_INTERVAL_EPOCHS = 1

# Model checkpoint retention - number of recent checkpoints to keep
CHECKPOINT_RETENTION_COUNT = 5


# =============================================================================
# MCTS Thresholds (December 2025)
# =============================================================================

# MCTS search timeout per move (seconds) - 0 means no timeout, budget-driven
MCTS_SEARCH_TIMEOUT = 0

# Minimum visit count threshold before selecting move
MCTS_MIN_VISIT_THRESHOLD = 10

# Gumbel MCTS top-K actions to sample
GUMBEL_TOP_K = 16

# Gumbel visit completion coefficient
GUMBEL_C_VISIT = 50.0

# Gumbel exploration constant (UCB)
GUMBEL_C_PUCT = 1.5

# Gumbel MCTS budget constants (SOURCE OF TRUTH)
# Previously imported from gumbel_common.py, now defined here to avoid torch import
# gumbel_common.py should import FROM here, not the other way around
#
# December 2025: Increased STANDARD from 150 to 800 to match AlphaZero.
# Low budgets (150) produce weak training data that plateaus at ~1600 Elo.
# AlphaZero uses 800 simulations; this is the minimum for strong models.
#
# Budget tiers for different training phases:
# - THROUGHPUT (64): Fast bootstrap, curriculum warmup (< 1500 Elo)
# - STANDARD (800): Core training for 1500-1800 Elo
# - QUALITY (800): Evaluation and gauntlet games
# - ULTIMATE (1600): Strong benchmarks, 1800-2000 Elo
# - MASTER (3200): 2000+ Elo training, tournament-quality moves
GUMBEL_BUDGET_THROUGHPUT = 64    # Maximum speed, low quality (for fast iteration)
GUMBEL_BUDGET_STANDARD = 800     # Default for training selfplay (AlphaZero uses 800)
GUMBEL_BUDGET_QUALITY = 800      # High quality for evaluation/gauntlet
GUMBEL_BUDGET_ULTIMATE = 1600    # Maximum quality for final benchmarks
GUMBEL_BUDGET_MASTER = 3200      # 2000+ Elo training (Dec 2025: for breaking Elo plateau)
GUMBEL_DEFAULT_BUDGET = GUMBEL_BUDGET_STANDARD

# December 29, 2025: Bootstrap budget tiers for data-starved configs
# When a config has very few games, prioritize game generation speed over quality.
# This enables faster bootstrapping of new/starved configurations.
# Thresholds: game_count < threshold uses the corresponding budget
GUMBEL_BUDGET_BOOTSTRAP_TIER1 = 64   # For <100 games: maximum throughput
GUMBEL_BUDGET_BOOTSTRAP_TIER2 = 150  # For <500 games: medium speed
GUMBEL_BUDGET_BOOTSTRAP_TIER3 = 200  # For <1000 games: balanced
# Above 1000 games: use Elo-based adaptive budget (STANDARD/QUALITY/ULTIMATE/MASTER)

# Game count thresholds for bootstrap budget tiers
BOOTSTRAP_TIER1_GAME_THRESHOLD = 100   # Very starved
BOOTSTRAP_TIER2_GAME_THRESHOLD = 500   # Moderately starved
BOOTSTRAP_TIER3_GAME_THRESHOLD = 1000  # Somewhat starved


# =============================================================================
# GPU Thresholds (December 2025)
# =============================================================================

# Memory check interval for GPU monitoring (seconds)
GPU_MEMORY_CHECK_INTERVAL = 60

# Batch operation timeout for GPU operations (seconds)
GPU_BATCH_TIMEOUT = 300

# Device synchronization timeout (seconds)
GPU_DEVICE_SYNC_TIMEOUT = 30

# GPU warmup steps before enabling optimizations
GPU_WARMUP_STEPS = 10

# Minimum GPU memory free before allocation (GB)
GPU_MIN_FREE_MEMORY_GB = 2.0

# Neural network batch evaluation timeout (milliseconds)
NN_EVAL_BATCH_TIMEOUT_MS = 2


# =============================================================================
# Coordination Thresholds (December 2025)
# =============================================================================
# Note: Some of these duplicate P2P/cluster sections above for backward compatibility

# Daemon restart delay after failure (seconds)
DAEMON_RESTART_DELAY_BASE = 5

# Daemon restart exponential backoff cap (seconds)
DAEMON_RESTART_DELAY_MAX = 300

# Daemon restart count reset after stability period (seconds)
DAEMON_RESTART_RESET_AFTER = 3600

# Daemon lifecycle timeouts (December 2025)
DAEMON_JOIN_TIMEOUT = 5.0  # Thread/process join timeout
DAEMON_SHUTDOWN_TIMEOUT = 10.0  # Max time to wait for graceful shutdown
DAEMON_FORCE_KILL_TIMEOUT = 5.0  # Additional time after shutdown before force kill
DAEMON_READY_TIMEOUT = 30.0  # Max time to wait for daemon to be ready
MAX_DAEMON_RESTART_ATTEMPTS = 5  # Max restart attempts per daemon

# Health check and heartbeat (December 2025)
HEALTH_CHECK_TIMEOUT = 5  # Socket/HTTP health check timeout
HEALTH_CHECK_INTERVAL_SECONDS = 30  # Global health check interval
HEARTBEAT_INTERVAL_SECONDS = 30  # Daemon heartbeat interval
HEARTBEAT_TIMEOUT_SECONDS = 60  # Time before declaring daemon unresponsive
SQLITE_CONNECT_TIMEOUT = 5.0  # SQLite connection timeout

# Circuit breaker settings (December 2025)
CIRCUIT_BREAKER_TIMEOUT = 60.0  # Time before circuit breaker opens
CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT = 30.0  # Half-open testing period
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening circuit

# Recovery settings (December 2025)
RECOVERY_ATTEMPT_COOLDOWN_SECONDS = 300  # Cooldown between recovery attempts
CAPACITY_REFRESH_INTERVAL = 300.0  # Interval for refreshing cluster capacity
SYNC_STALL_DETECTION_TIMEOUT = 600  # Timeout before declaring sync stalled

# Lock timeout for coordination primitives (seconds)
COORDINATION_LOCK_TIMEOUT = 60

# Queue poll interval for work queue checks (seconds)
QUEUE_POLL_INTERVAL = 5

# Event handler timeout (seconds)
EVENT_HANDLER_TIMEOUT = 600

# Work timeout for async tasks (seconds)
WORK_TIMEOUT = 300

# Lock acquisition timeout for distributed locks (seconds)
LOCK_ACQUIRE_TIMEOUT = 60


# =============================================================================
# Distributed Transport Thresholds (December 2025)
# =============================================================================

# Sync timeout for distributed data synchronization (seconds)
DISTRIBUTED_SYNC_TIMEOUT = 300

# Retry backoff multiplier for exponential backoff
RETRY_BACKOFF_MULTIPLIER = 2.0

# Retry base delay (seconds)
RETRY_BASE_DELAY_SECONDS = 1.0

# Maximum retry attempts for transient failures
MAX_RETRY_ATTEMPTS_DISTRIBUTED = 3

# SSH maximum retries per connection attempt
SSH_MAX_RETRIES = 2

# Transport health: consecutive failures before disabling transport
TRANSPORT_FAILURE_THRESHOLD = 3

# Transport health: disable duration after failures (seconds)
TRANSPORT_DISABLE_DURATION = 300

# Transport health: latency history weight (exponential moving average)
TRANSPORT_LATENCY_WEIGHT = 0.7

# Transport health: minimum samples before preferring a transport
TRANSPORT_MIN_SAMPLES_FOR_PREFERENCE = 3


# =============================================================================
# Alert & Monitoring Thresholds (December 2025)
# =============================================================================

# Minimum interval between duplicate alerts (seconds)
MIN_ALERT_INTERVAL_SECONDS = 1800  # 30 minutes

# Maximum alerts per hour (rate limiting)
MAX_ALERTS_PER_HOUR = 20

# Partition threshold - suppress if >50% nodes have same issue
ALERT_PARTITION_THRESHOLD = 0.5


# =============================================================================
# Selfplay & Game Execution Thresholds (December 2025)
# =============================================================================

# Maximum moves per game before forced termination
MAX_MOVES_PER_GAME = 10000

# Game execution timeout (seconds) - 0 means no timeout
GAME_EXECUTION_TIMEOUT = 0

# Minimum game length to be considered valid (moves)
MIN_VALID_GAME_LENGTH = 5

# Maximum game duration before termination (seconds)
MAX_GAME_DURATION_SECONDS = 600

# Minimum games per hour for selfplay monitoring
MIN_SELFPLAY_GAMES_PER_HOUR = 100


# =============================================================================
# Game Recording Quality Thresholds (December 2025)
# =============================================================================

# Pre-recording validation gates
RECORDING_MIN_MOVES = 5  # Minimum moves to record a game
RECORDING_MAX_MOVES = 500  # Maximum moves before suspicious
RECORDING_REQUIRE_VICTORY_TYPE = True  # Require canonical victory type
RECORDING_REQUIRE_COMPLETED_STATUS = True  # Require completed game status

# Quality-based sync filtering
SYNC_MIN_QUALITY = 0.5  # Skip syncing databases with avg quality < 50%
SYNC_QUALITY_SAMPLE_SIZE = 20  # Sample this many recent games for quality check

# Data cleanup thresholds
CLEANUP_QUALITY_THRESHOLD_DELETE = 0.1  # Delete if avg quality < 10%
CLEANUP_QUALITY_THRESHOLD_QUARANTINE = 0.3  # Quarantine if avg quality < 30%
CLEANUP_MOVE_COVERAGE_THRESHOLD = 0.1  # Quarantine if <10% have move data
CLEANUP_MIN_GAMES_BEFORE_DELETE = 100  # Don't delete DBs with few games
CLEANUP_SCAN_INTERVAL_SECONDS = 3600  # 1 hour between scans


# =============================================================================
# Helper Functions
# =============================================================================


def get_gumbel_budget(difficulty: int = 6) -> int:
    """Get Gumbel MCTS budget for a difficulty level.

    Args:
        difficulty: Difficulty level 1-11

    Returns:
        Recommended simulation budget
    """
    if difficulty <= 6:
        return GUMBEL_BUDGET_STANDARD
    elif difficulty <= 9:
        return GUMBEL_BUDGET_QUALITY
    else:
        return GUMBEL_BUDGET_ULTIMATE
