"""Unified Threshold Constants for RingRift AI Service.

This module provides SINGLE SOURCE OF TRUTH for all threshold values
used across training, evaluation, promotion, and rollback systems.

Import these constants instead of hardcoding values:

    from app.config.thresholds import (
        TRAINING_TRIGGER_GAMES,
        ELO_DROP_ROLLBACK,
        ELO_IMPROVEMENT_PROMOTE,
    )

See docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

# =============================================================================
# Training Thresholds
# =============================================================================

# Games needed to trigger training (per config)
TRAINING_TRIGGER_GAMES = 500

# Minimum interval between training runs (seconds)
TRAINING_MIN_INTERVAL_SECONDS = 1200  # 20 minutes

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

# Cooldown between promotion attempts (seconds)
PROMOTION_COOLDOWN_SECONDS = 900  # 15 minutes

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
