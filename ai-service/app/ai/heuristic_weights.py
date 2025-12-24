"""Heuristic weight profiles for RingRift.

This module centralises all scalar weights used by :class:`HeuristicAI` for
position evaluation, and exposes named, versioned profiles that can be
referenced from both runtime configs (``AIConfig.heuristic_profile_id``) and
offline training/tuning tools.

Design goals:

* Keep a **single source of truth** for heuristic weights instead of
  scattering literals across the evaluator.
* Support **versioned base profiles** (e.g. ``heuristic_v1_balanced``)
  backed by RingRift-specific signals described in ``AI_ARCHITECTURE.md``.
* Allow lightweight **personas** (aggressive / territorial / defensive)
  implemented as small deltas over the balanced profile.
* Remain JSON-serialisable so training pipelines can snapshot and tune
  profiles without impacting runtime defaults.

The keys in each profile intentionally mirror the attribute names on
:class:`HeuristicAI` (``WEIGHT_STACK_CONTROL``, ``WEIGHT_TERRITORY``, etc.)
so that instances can simply ``setattr(self, name, value)`` when applying a
profile.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

HeuristicWeights = dict[str, float]


# --- v1 Balanced Base Profile ----------------------------------------------
#
# These weights are the validated production defaults for higher difficulty
# heuristic AI (D4+). Validated 2025-12-24 with 128 games/opponent:
# - Aggregate fitness: 0.551 (55.1%)
# - Win rates: 53-60% vs all opponent types (balanced, aggressive, territorial, defensive)
#
# These round-number weights empirically outperform CMA-ES optimized weights
# (which achieved only 0.464 fitness). The simpler values appear more robust
# across opponent styles.
#
# The *shape* of the profile and the semantic meaning of each key remain
# stable for compatibility with training pipelines and TS parity.

BASE_V1_BALANCED_WEIGHTS: HeuristicWeights = {
    "WEIGHT_STACK_CONTROL": 10.0,
    "WEIGHT_STACK_HEIGHT": 5.0,
    "WEIGHT_CAP_HEIGHT": 6.0,  # v1.5: Summed cap height (capture power dominance)
    "WEIGHT_TERRITORY": 8.0,
    "WEIGHT_RINGS_IN_HAND": 3.0,
    "WEIGHT_CENTER_CONTROL": 4.0,
    "WEIGHT_ADJACENCY": 2.0,
    "WEIGHT_OPPONENT_THREAT": 6.0,
    "WEIGHT_MOBILITY": 4.0,
    "WEIGHT_ELIMINATED_RINGS": 12.0,
    "WEIGHT_LINE_POTENTIAL": 7.0,
    "WEIGHT_VICTORY_PROXIMITY": 20.0,
    "WEIGHT_MARKER_COUNT": 1.5,
    "WEIGHT_VULNERABILITY": 8.0,
    "WEIGHT_OVERTAKE_POTENTIAL": 8.0,
    "WEIGHT_TERRITORY_CLOSURE": 10.0,
    "WEIGHT_LINE_CONNECTIVITY": 6.0,
    "WEIGHT_TERRITORY_SAFETY": 5.0,
    "WEIGHT_STACK_MOBILITY": 4.0,
    # High-signal heuristic extensions (v1).
    "WEIGHT_OPPONENT_VICTORY_THREAT": 6.0,
    "WEIGHT_FORCED_ELIMINATION_RISK": 4.0,
    "WEIGHT_LPS_ACTION_ADVANTAGE": 2.0,
    "WEIGHT_MULTI_LEADER_THREAT": 2.0,
    # v1.1 refactor: previously hardcoded penalties/bonuses now configurable
    "WEIGHT_NO_STACKS_PENALTY": 50.0,
    "WEIGHT_SINGLE_STACK_PENALTY": 10.0,
    "WEIGHT_STACK_DIVERSITY_BONUS": 2.0,
    "WEIGHT_SAFE_MOVE_BONUS": 1.0,
    "WEIGHT_NO_SAFE_MOVES_PENALTY": 2.0,
    "WEIGHT_VICTORY_THRESHOLD_BONUS": 1000.0,
    "WEIGHT_RINGS_PROXIMITY_FACTOR": 50.0,
    "WEIGHT_TERRITORY_PROXIMITY_FACTOR": 50.0,
    "WEIGHT_TWO_IN_ROW": 1.0,
    "WEIGHT_THREE_IN_ROW": 2.0,
    "WEIGHT_FOUR_IN_ROW": 5.0,
    "WEIGHT_CONNECTED_NEIGHBOR": 1.0,
    "WEIGHT_GAP_POTENTIAL": 0.5,
    "WEIGHT_BLOCKED_STACK_PENALTY": 5.0,
    # v1.2: Swap (pie rule) opening evaluation - rewards swapping into
    # advantageous P1 openings (center control, strong positions)
    "WEIGHT_SWAP_OPENING_CENTER": 15.0,     # Bonus per P1 stack in center
    "WEIGHT_SWAP_OPENING_ADJACENCY": 3.0,   # Bonus for P1 stacks near center
    "WEIGHT_SWAP_OPENING_HEIGHT": 2.0,      # Bonus per stack height on P1 stacks
    # v1.3: Enhanced swap evaluation - Opening Position Classifier weights
    "WEIGHT_SWAP_CORNER_PENALTY": 8.0,      # Penalty for corner positions (weak)
    "WEIGHT_SWAP_EDGE_BONUS": 2.0,          # Bonus for edge positions (moderate)
    "WEIGHT_SWAP_DIAGONAL_BONUS": 6.0,      # Bonus for key diagonal positions
    "WEIGHT_SWAP_OPENING_STRENGTH": 20.0,   # Multiplier for normalized strength
    # v1.4: Training diversity - Swap decision randomness
    "WEIGHT_SWAP_EXPLORATION_TEMPERATURE": 0.0,  # Deterministic swaps
    # v1.5: Recovery evaluation weights
    "WEIGHT_RECOVERY_POTENTIAL": 6.0,       # Value of having recovery available
    "WEIGHT_RECOVERY_ELIGIBILITY": 8.0,     # Bonus/penalty for recovery eligibility status
    "WEIGHT_BURIED_RING_VALUE": 3.0,        # Value of buried rings as recovery resource
    "WEIGHT_RECOVERY_THREAT": 5.0,          # Threat from opponent's recovery potential
}


# Canonical ordered list of heuristic weight keys used by :class:`HeuristicAI`.
# This order is used by optimisation/diagnostics tooling (CMA-ES, GA,
# axis-aligned scans, etc.) and must remain in lockstep with the keys and
# insertion order of :data:`BASE_V1_BALANCED_WEIGHTS`. Tests assert this
# invariant so external tools can safely rely on it.
HEURISTIC_WEIGHT_KEYS: list[str] = [
    "WEIGHT_STACK_CONTROL",
    "WEIGHT_STACK_HEIGHT",
    "WEIGHT_CAP_HEIGHT",  # v1.5: Summed cap height (capture power dominance)
    "WEIGHT_TERRITORY",
    "WEIGHT_RINGS_IN_HAND",
    "WEIGHT_CENTER_CONTROL",
    "WEIGHT_ADJACENCY",
    "WEIGHT_OPPONENT_THREAT",
    "WEIGHT_MOBILITY",
    "WEIGHT_ELIMINATED_RINGS",
    "WEIGHT_LINE_POTENTIAL",
    "WEIGHT_VICTORY_PROXIMITY",
    "WEIGHT_MARKER_COUNT",
    "WEIGHT_VULNERABILITY",
    "WEIGHT_OVERTAKE_POTENTIAL",
    "WEIGHT_TERRITORY_CLOSURE",
    "WEIGHT_LINE_CONNECTIVITY",
    "WEIGHT_TERRITORY_SAFETY",
    "WEIGHT_STACK_MOBILITY",
    "WEIGHT_OPPONENT_VICTORY_THREAT",
    "WEIGHT_FORCED_ELIMINATION_RISK",
    "WEIGHT_LPS_ACTION_ADVANTAGE",
    "WEIGHT_MULTI_LEADER_THREAT",
    # v1.1 refactor: previously hardcoded penalties/bonuses
    "WEIGHT_NO_STACKS_PENALTY",
    "WEIGHT_SINGLE_STACK_PENALTY",
    "WEIGHT_STACK_DIVERSITY_BONUS",
    "WEIGHT_SAFE_MOVE_BONUS",
    "WEIGHT_NO_SAFE_MOVES_PENALTY",
    "WEIGHT_VICTORY_THRESHOLD_BONUS",
    "WEIGHT_RINGS_PROXIMITY_FACTOR",
    "WEIGHT_TERRITORY_PROXIMITY_FACTOR",
    "WEIGHT_TWO_IN_ROW",
    "WEIGHT_THREE_IN_ROW",
    "WEIGHT_FOUR_IN_ROW",
    "WEIGHT_CONNECTED_NEIGHBOR",
    "WEIGHT_GAP_POTENTIAL",
    "WEIGHT_BLOCKED_STACK_PENALTY",
    # v1.2: Swap (pie rule) opening evaluation
    "WEIGHT_SWAP_OPENING_CENTER",
    "WEIGHT_SWAP_OPENING_ADJACENCY",
    "WEIGHT_SWAP_OPENING_HEIGHT",
    # v1.3: Enhanced swap evaluation - Opening Position Classifier
    "WEIGHT_SWAP_CORNER_PENALTY",
    "WEIGHT_SWAP_EDGE_BONUS",
    "WEIGHT_SWAP_DIAGONAL_BONUS",
    "WEIGHT_SWAP_OPENING_STRENGTH",
    # v1.4: Training diversity - Swap decision randomness
    "WEIGHT_SWAP_EXPLORATION_TEMPERATURE",
    # v1.5: Recovery evaluation weights
    "WEIGHT_RECOVERY_POTENTIAL",
    "WEIGHT_RECOVERY_ELIGIBILITY",
    "WEIGHT_BURIED_RING_VALUE",
    "WEIGHT_RECOVERY_THREAT",
]


def _with_deltas(
    base: Mapping[str, float],
    *,
    scale: Mapping[str, float] | None = None,
    offset: Mapping[str, float] | None = None,
) -> HeuristicWeights:
    """Create a new profile from *base* by applying per-key scale/offset.

    This is deliberately simple: it is intended for designer-friendly persona
    tweaks (e.g. +20% capture emphasis, -20% territory safety) rather than
    for gradient-based optimisation. All keys in ``base`` are preserved so
    that profiles remain structurally compatible.
    """

    scale = scale or {}
    offset = offset or {}
    out: HeuristicWeights = {}
    for key, value in base.items():
        s = scale.get(key, 1.0)
        o = offset.get(key, 0.0)
        out[key] = value * s + o
    return out


# --- v1 Personas -----------------------------------------------------------
#
# These personas are defined as small, interpretable adjustments around the
# balanced profile. They are intentionally conservative so as not to break
# existing expectations while still giving designers room to experiment.
# TS personas must mirror these deltas; see
# src/shared/engine/heuristicEvaluation.ts.

# Balanced: identical to the base profile.
HEURISTIC_V1_BALANCED: HeuristicWeights = dict(BASE_V1_BALANCED_WEIGHTS)

# Aggressive: emphasise capture/elimination and short-term victory progress,
# dial back territory safety and vulnerability penalties slightly so the AI
# is more willing to take calculated risks.
HEURISTIC_V1_AGGRESSIVE: HeuristicWeights = _with_deltas(
    BASE_V1_BALANCED_WEIGHTS,
    scale={
        "WEIGHT_ELIMINATED_RINGS": 1.25,
        "WEIGHT_OVERTAKE_POTENTIAL": 1.25,
        "WEIGHT_VICTORY_PROXIMITY": 1.15,
        "WEIGHT_LINE_POTENTIAL": 1.1,
        # Slightly downweight safety concerns
        "WEIGHT_VULNERABILITY": 0.85,
        "WEIGHT_TERRITORY_SAFETY": 0.85,
        # High-signal extensions
        "WEIGHT_OPPONENT_VICTORY_THREAT": 1.10,
        "WEIGHT_FORCED_ELIMINATION_RISK": 0.85,
        "WEIGHT_LPS_ACTION_ADVANTAGE": 1.0,
        "WEIGHT_MULTI_LEADER_THREAT": 0.8,
    },
)

# Territorial: emphasise territory ownership/closure and line structure,
# de-emphasise raw elimination so the AI is more likely to play for space and
# long-term region control.
HEURISTIC_V1_TERRITORIAL: HeuristicWeights = _with_deltas(
    BASE_V1_BALANCED_WEIGHTS,
    scale={
        "WEIGHT_TERRITORY": 1.25,
        "WEIGHT_TERRITORY_CLOSURE": 1.25,
        "WEIGHT_TERRITORY_SAFETY": 1.2,
        "WEIGHT_LINE_POTENTIAL": 1.15,
        "WEIGHT_LINE_CONNECTIVITY": 1.15,
        "WEIGHT_MARKER_COUNT": 1.1,
        # Slightly downweight pure elimination focus
        "WEIGHT_ELIMINATED_RINGS": 0.9,
        # High-signal extensions
        "WEIGHT_OPPONENT_VICTORY_THREAT": 1.15,
        "WEIGHT_FORCED_ELIMINATION_RISK": 1.0,
        "WEIGHT_LPS_ACTION_ADVANTAGE": 1.1,
        "WEIGHT_MULTI_LEADER_THREAT": 1.25,
    },
)

# Defensive/control: prioritise stack safety, vulnerability awareness, and
# general mobility; slightly reduce aggression so the AI avoids speculative
# captures that leave it exposed.
HEURISTIC_V1_DEFENSIVE: HeuristicWeights = _with_deltas(
    BASE_V1_BALANCED_WEIGHTS,
    scale={
        "WEIGHT_STACK_CONTROL": 1.15,
        "WEIGHT_STACK_MOBILITY": 1.2,
        "WEIGHT_MOBILITY": 1.1,
        "WEIGHT_VULNERABILITY": 1.25,
        "WEIGHT_TERRITORY_SAFETY": 1.15,
        # Tone down overtake/elimination eagerness a bit
        "WEIGHT_OVERTAKE_POTENTIAL": 0.9,
        "WEIGHT_ELIMINATED_RINGS": 0.9,
        # High-signal extensions
        "WEIGHT_OPPONENT_VICTORY_THREAT": 1.05,
        "WEIGHT_FORCED_ELIMINATION_RISK": 1.3,
        "WEIGHT_LPS_ACTION_ADVANTAGE": 1.2,
        "WEIGHT_MULTI_LEADER_THREAT": 1.1,
    },
)


# --- Difficulty-scaled Profiles for Pure Heuristic AI -----------------------
#
# These profiles are designed for pure heuristic play (no search algorithm).
# Each difficulty level has genuinely different weights to create distinct
# play styles and justify separate Elo tracking.
#
# D2 (Weak): 50% scaled weights - plays poorly, misses key features
# D3 (Medium): 75% scaled - reasonable but suboptimal play
# D4 (Strong): 100% balanced - full strength heuristic
# D5 (Expert): Aggressive profile - tactical, optimized play

def _scale_all_weights(base: dict[str, float], scale: float) -> HeuristicWeights:
    """Scale all weights uniformly by a factor."""
    return {k: v * scale for k, v in base.items()}


# D2: Weak heuristic - 50% of balanced weights
# Plays poorly because it undervalues most features
HEURISTIC_V1_WEAK: HeuristicWeights = _scale_all_weights(BASE_V1_BALANCED_WEIGHTS, 0.5)

# D3: Medium heuristic - 75% of balanced weights
# Reasonable but suboptimal play
HEURISTIC_V1_MEDIUM: HeuristicWeights = _scale_all_weights(BASE_V1_BALANCED_WEIGHTS, 0.75)

# D4: Strong heuristic - 100% balanced (same as BALANCED)
HEURISTIC_V1_STRONG: HeuristicWeights = dict(BASE_V1_BALANCED_WEIGHTS)

# D5: Expert heuristic - uses AGGRESSIVE profile for tactical play
HEURISTIC_V1_EXPERT: HeuristicWeights = dict(HEURISTIC_V1_AGGRESSIVE)


# --- Board-type-specific Optimized Profiles ---------------------------------
#
# Larger boards (19x19) have fundamentally different strategic dynamics
# compared to smaller boards (8x8). These profiles were optimized separately.

# Square19 2-player optimized weights (CMA-ES, 15 generations, fitness: 83.3%)
# Key differences from square8:
# - CENTER_CONTROL: ~0 on 8x8 -> 4.45 on 19x19 (center is critical on big board)
# - MARKER_COUNT: +2.16 on 8x8 -> -1.90 on 19x19 (too many markers = overextension)
# - OVERTAKE_POTENTIAL: 3.75 on 8x8 -> 8.34 on 19x19 (captures more important)
# - STACK_DIVERSITY_BONUS: -1.33 on 8x8 -> +2.96 on 19x19 (spread out on big board)
# - TWO_IN_ROW: 4.13 on 8x8 -> 0.28 on 19x19 (early lines less useful)
HEURISTIC_V1_SQUARE19_2P: HeuristicWeights = {
    "WEIGHT_STACK_CONTROL": 10.57,
    "WEIGHT_STACK_HEIGHT": 4.52,
    "WEIGHT_CAP_HEIGHT": 5.19,
    "WEIGHT_TERRITORY": 7.24,
    "WEIGHT_RINGS_IN_HAND": 1.53,
    "WEIGHT_CENTER_CONTROL": 4.45,  # Center is CRITICAL on 19x19
    "WEIGHT_ADJACENCY": 0.81,
    "WEIGHT_OPPONENT_THREAT": 6.05,
    "WEIGHT_MOBILITY": 3.48,
    "WEIGHT_ELIMINATED_RINGS": 11.91,
    "WEIGHT_LINE_POTENTIAL": 8.37,  # Long-term lines matter more
    "WEIGHT_VICTORY_PROXIMITY": 20.43,
    "WEIGHT_MARKER_COUNT": -1.90,  # NEGATIVE: too many markers is bad on big board
    "WEIGHT_VULNERABILITY": 7.68,
    "WEIGHT_OVERTAKE_POTENTIAL": 8.34,  # Captures more important on big board
    "WEIGHT_TERRITORY_CLOSURE": 5.67,
    "WEIGHT_LINE_CONNECTIVITY": 4.63,
    "WEIGHT_TERRITORY_SAFETY": 4.41,
    "WEIGHT_STACK_MOBILITY": 3.79,  # Movement flexibility key
    "WEIGHT_OPPONENT_VICTORY_THREAT": 5.44,  # More defensive awareness
    "WEIGHT_FORCED_ELIMINATION_RISK": 3.87,  # Much more risk-aware
    "WEIGHT_LPS_ACTION_ADVANTAGE": 1.73,
    "WEIGHT_MULTI_LEADER_THREAT": 1.73,
    "WEIGHT_NO_STACKS_PENALTY": 50.93,
    "WEIGHT_SINGLE_STACK_PENALTY": 9.20,
    "WEIGHT_STACK_DIVERSITY_BONUS": 2.96,  # Spread out on big board
    "WEIGHT_SAFE_MOVE_BONUS": 1.20,
    "WEIGHT_NO_SAFE_MOVES_PENALTY": 1.33,
    "WEIGHT_VICTORY_THRESHOLD_BONUS": 999.42,
    "WEIGHT_RINGS_PROXIMITY_FACTOR": 50.51,
    "WEIGHT_TERRITORY_PROXIMITY_FACTOR": 49.73,
    "WEIGHT_TWO_IN_ROW": 0.28,  # Early lines less useful on big board
    "WEIGHT_THREE_IN_ROW": 2.01,
    "WEIGHT_FOUR_IN_ROW": 5.80,
    "WEIGHT_CONNECTED_NEIGHBOR": 0.99,
    "WEIGHT_GAP_POTENTIAL": -0.26,
    "WEIGHT_BLOCKED_STACK_PENALTY": 5.80,
    "WEIGHT_SWAP_OPENING_CENTER": 15.60,
    "WEIGHT_SWAP_OPENING_ADJACENCY": 3.17,
    "WEIGHT_SWAP_OPENING_HEIGHT": 1.22,
    "WEIGHT_SWAP_CORNER_PENALTY": 7.27,
    "WEIGHT_SWAP_EDGE_BONUS": 2.15,
    "WEIGHT_SWAP_DIAGONAL_BONUS": 5.19,
    "WEIGHT_SWAP_OPENING_STRENGTH": 21.10,
    "WEIGHT_SWAP_EXPLORATION_TEMPERATURE": -0.38,
    # v1.5: Recovery evaluation weights (larger board = recovery more strategic)
    "WEIGHT_RECOVERY_POTENTIAL": 8.0,
    "WEIGHT_RECOVERY_ELIGIBILITY": 10.0,
    "WEIGHT_BURIED_RING_VALUE": 4.0,
    "WEIGHT_RECOVERY_THREAT": 6.0,
}


# --- Player-count-specific Optimized Profiles -------------------------------
#
# These profiles were optimized via CMA-ES for specific player counts.
# Multi-player games have different strategic dynamics that require
# different weight balances.

# 3-player optimized weights (CMA-ES, 20 generations, fitness: 65%)
# Key differences from 2-player:
# - Higher MOBILITY (5.54 vs 4.35) - movement flexibility more critical
# - Higher LPS_ACTION_ADVANTAGE (2.77 vs 0.73) - LPS scoring matters more
# - Higher GAP_POTENTIAL (1.39 vs -0.43) - gaps become strategic assets
# - Lower MARKER_COUNT (0.17 vs 3.38) - marker density less important
# - Positive SWAP_EXPLORATION_TEMPERATURE (0.81) - more swap variety helps
HEURISTIC_V1_3P: HeuristicWeights = {
    "WEIGHT_STACK_CONTROL": 9.84,
    "WEIGHT_STACK_HEIGHT": 5.80,
    "WEIGHT_CAP_HEIGHT": 5.58,
    "WEIGHT_TERRITORY": 8.40,
    "WEIGHT_RINGS_IN_HAND": 1.63,
    "WEIGHT_CENTER_CONTROL": 4.13,
    "WEIGHT_ADJACENCY": 0.88,
    "WEIGHT_OPPONENT_THREAT": 5.29,
    "WEIGHT_MOBILITY": 5.54,
    "WEIGHT_ELIMINATED_RINGS": 10.89,
    "WEIGHT_LINE_POTENTIAL": 7.13,
    "WEIGHT_VICTORY_PROXIMITY": 20.67,
    "WEIGHT_MARKER_COUNT": 0.17,
    "WEIGHT_VULNERABILITY": 7.28,
    "WEIGHT_OVERTAKE_POTENTIAL": 7.53,
    "WEIGHT_TERRITORY_CLOSURE": 8.61,
    "WEIGHT_LINE_CONNECTIVITY": 3.62,
    "WEIGHT_TERRITORY_SAFETY": 5.06,
    "WEIGHT_STACK_MOBILITY": 3.04,
    "WEIGHT_OPPONENT_VICTORY_THREAT": 6.34,
    "WEIGHT_FORCED_ELIMINATION_RISK": 3.05,
    "WEIGHT_LPS_ACTION_ADVANTAGE": 2.77,
    "WEIGHT_MULTI_LEADER_THREAT": 1.97,
    "WEIGHT_NO_STACKS_PENALTY": 49.49,
    "WEIGHT_SINGLE_STACK_PENALTY": 10.60,
    "WEIGHT_STACK_DIVERSITY_BONUS": 1.62,
    "WEIGHT_SAFE_MOVE_BONUS": 0.93,
    "WEIGHT_NO_SAFE_MOVES_PENALTY": 2.60,
    "WEIGHT_VICTORY_THRESHOLD_BONUS": 1000.80,
    "WEIGHT_RINGS_PROXIMITY_FACTOR": 49.70,
    "WEIGHT_TERRITORY_PROXIMITY_FACTOR": 50.68,
    "WEIGHT_TWO_IN_ROW": 0.79,
    "WEIGHT_THREE_IN_ROW": 1.13,
    "WEIGHT_FOUR_IN_ROW": 5.32,
    "WEIGHT_CONNECTED_NEIGHBOR": 0.42,
    "WEIGHT_GAP_POTENTIAL": 1.39,
    "WEIGHT_BLOCKED_STACK_PENALTY": 5.70,
    "WEIGHT_SWAP_OPENING_CENTER": 15.56,
    "WEIGHT_SWAP_OPENING_ADJACENCY": 2.69,
    "WEIGHT_SWAP_OPENING_HEIGHT": 2.23,
    "WEIGHT_SWAP_CORNER_PENALTY": 7.68,
    "WEIGHT_SWAP_EDGE_BONUS": 1.38,
    "WEIGHT_SWAP_DIAGONAL_BONUS": 6.64,
    "WEIGHT_SWAP_OPENING_STRENGTH": 19.60,
    "WEIGHT_SWAP_EXPLORATION_TEMPERATURE": 0.81,
    # v1.5: Recovery evaluation weights (multi-player = recovery more tactical)
    "WEIGHT_RECOVERY_POTENTIAL": 7.0,
    "WEIGHT_RECOVERY_ELIGIBILITY": 9.0,
    "WEIGHT_BURIED_RING_VALUE": 3.5,
    "WEIGHT_RECOVERY_THREAT": 5.5,
}

# 4-player optimized weights (CMA-ES, 20 generations, fitness: 75%)
# Key differences from 2-player:
# - Higher CENTER_CONTROL (5.35 vs 1.81) - board center is critical
# - Negative ADJACENCY (-0.98) - clustering is penalized (spread out)
# - Negative MARKER_COUNT (-0.42) - too many markers is bad
# - Lower MOBILITY (2.73 vs 4.35) - positional stability over movement
# - Higher STACK_DIVERSITY_BONUS (1.99 vs 0.03) - diversification helps
# - Near-zero SAFE_MOVE_BONUS (0.02) - can't play too safely
HEURISTIC_V1_4P: HeuristicWeights = {
    "WEIGHT_STACK_CONTROL": 10.21,
    "WEIGHT_STACK_HEIGHT": 6.85,
    "WEIGHT_CAP_HEIGHT": 5.64,
    "WEIGHT_TERRITORY": 6.55,
    "WEIGHT_RINGS_IN_HAND": 0.25,
    "WEIGHT_CENTER_CONTROL": 5.35,
    "WEIGHT_ADJACENCY": -0.98,
    "WEIGHT_OPPONENT_THREAT": 4.28,
    "WEIGHT_MOBILITY": 2.73,
    "WEIGHT_ELIMINATED_RINGS": 11.67,
    "WEIGHT_LINE_POTENTIAL": 7.16,
    "WEIGHT_VICTORY_PROXIMITY": 19.91,
    "WEIGHT_MARKER_COUNT": -0.42,
    "WEIGHT_VULNERABILITY": 7.69,
    "WEIGHT_OVERTAKE_POTENTIAL": 7.27,
    "WEIGHT_TERRITORY_CLOSURE": 6.13,
    "WEIGHT_LINE_CONNECTIVITY": 3.09,
    "WEIGHT_TERRITORY_SAFETY": 4.50,
    "WEIGHT_STACK_MOBILITY": 3.07,
    "WEIGHT_OPPONENT_VICTORY_THREAT": 4.19,
    "WEIGHT_FORCED_ELIMINATION_RISK": 4.99,
    "WEIGHT_LPS_ACTION_ADVANTAGE": 1.41,
    "WEIGHT_MULTI_LEADER_THREAT": 1.65,
    "WEIGHT_NO_STACKS_PENALTY": 50.68,
    "WEIGHT_SINGLE_STACK_PENALTY": 9.50,
    "WEIGHT_STACK_DIVERSITY_BONUS": 1.99,
    "WEIGHT_SAFE_MOVE_BONUS": 0.02,
    "WEIGHT_NO_SAFE_MOVES_PENALTY": 1.35,
    "WEIGHT_VICTORY_THRESHOLD_BONUS": 1001.28,
    "WEIGHT_RINGS_PROXIMITY_FACTOR": 50.98,
    "WEIGHT_TERRITORY_PROXIMITY_FACTOR": 50.06,
    "WEIGHT_TWO_IN_ROW": 0.20,
    "WEIGHT_THREE_IN_ROW": 1.53,
    "WEIGHT_FOUR_IN_ROW": 5.64,
    "WEIGHT_CONNECTED_NEIGHBOR": 1.06,
    "WEIGHT_GAP_POTENTIAL": 0.23,
    "WEIGHT_BLOCKED_STACK_PENALTY": 5.13,
    "WEIGHT_SWAP_OPENING_CENTER": 15.32,
    "WEIGHT_SWAP_OPENING_ADJACENCY": 3.24,
    "WEIGHT_SWAP_OPENING_HEIGHT": 1.40,
    "WEIGHT_SWAP_CORNER_PENALTY": 5.67,
    "WEIGHT_SWAP_EDGE_BONUS": 2.27,
    "WEIGHT_SWAP_DIAGONAL_BONUS": 5.71,
    "WEIGHT_SWAP_OPENING_STRENGTH": 20.96,
    "WEIGHT_SWAP_EXPLORATION_TEMPERATURE": -0.19,
    # v1.5: Recovery evaluation weights (4P = recovery is critical for survival)
    "WEIGHT_RECOVERY_POTENTIAL": 8.0,
    "WEIGHT_RECOVERY_ELIGIBILITY": 10.0,
    "WEIGHT_BURIED_RING_VALUE": 4.0,
    "WEIGHT_RECOVERY_THREAT": 6.0,
}


# --- Registry --------------------------------------------------------------
#
# Public mapping from profile_id -> HeuristicWeights. The keys here are what
# ``AIConfig.heuristic_profile_id`` and the canonical difficulty ladder refer
# to. We include both high-level persona ids and the ladder-oriented ids used
# today so that existing configs/tests continue to work unchanged.
#
# Profile key naming convention:
#   - Board×player specific: "heuristic_v1_{board}_{n}p" (e.g., "heuristic_v1_sq8_2p")
#   - Legacy player-only: "heuristic_v1_{n}p" (aliases to sq8 profiles for backwards compat)
#
# Board abbreviations: sq8 = square8, sq19 = square19, hex = hexagonal

HEURISTIC_WEIGHT_PROFILES: dict[str, HeuristicWeights] = {
    # High-level, persona-oriented ids (preferred for new configs).
    "heuristic_v1_balanced": HEURISTIC_V1_BALANCED,
    "heuristic_v1_aggressive": HEURISTIC_V1_AGGRESSIVE,
    "heuristic_v1_territorial": HEURISTIC_V1_TERRITORIAL,
    "heuristic_v1_defensive": HEURISTIC_V1_DEFENSIVE,
    #
    # === Board × Player Matrix (9 combinations) ===
    # These are the canonical profile keys for CMA-ES training.
    # Each board×player combination gets its own optimized weights.
    #
    # Square8 profiles (primary training board)
    "heuristic_v1_sq8_2p": HEURISTIC_V1_BALANCED,   # square8 2p (baseline)
    "heuristic_v1_sq8_3p": HEURISTIC_V1_3P,         # square8 3p (CMA-ES optimized)
    "heuristic_v1_sq8_4p": HEURISTIC_V1_4P,         # square8 4p (CMA-ES optimized)
    #
    # Square19 profiles (larger board, different strategic dynamics)
    "heuristic_v1_sq19_2p": HEURISTIC_V1_SQUARE19_2P,  # square19 2p (CMA-ES optimized)
    "heuristic_v1_sq19_3p": HEURISTIC_V1_3P,           # square19 3p (fallback to 3p base)
    "heuristic_v1_sq19_4p": HEURISTIC_V1_4P,           # square19 4p (fallback to 4p base)
    #
    # Hexagonal profiles (different topology)
    "heuristic_v1_hex_2p": HEURISTIC_V1_BALANCED,   # hex 2p (fallback to balanced)
    "heuristic_v1_hex_3p": HEURISTIC_V1_3P,         # hex 3p (fallback to 3p base)
    "heuristic_v1_hex_4p": HEURISTIC_V1_4P,         # hex 4p (fallback to 4p base)
    #
    # === Legacy player-count-only profiles (backwards compatibility) ===
    # These alias to the corresponding square8 profiles.
    # Use get_weights_for_board() for new code to get board-specific weights.
    "heuristic_v1_2p": HEURISTIC_V1_BALANCED,  # Alias -> sq8_2p
    "heuristic_v1_3p": HEURISTIC_V1_3P,        # Alias -> sq8_3p
    "heuristic_v1_4p": HEURISTIC_V1_4P,        # Alias -> sq8_4p
    #
    # Legacy board-specific key (long form, deprecated - use sq19_2p)
    "heuristic_v1_square19_2p": HEURISTIC_V1_SQUARE19_2P,
    #
    # Optimized ensemble profile (average of player-specific trained weights).
    # Falls back to balanced until trained weights are loaded.
    "heuristic_v1_optimized": HEURISTIC_V1_BALANCED,
    #
    # === Difficulty-scaled Pure Heuristic Profiles ===
    # Each difficulty has genuinely different weights for distinct play styles.
    # These are for PURE heuristic AI (no search algorithm).
    "heuristic_v1_weak": HEURISTIC_V1_WEAK,      # D2: 50% scaled
    "heuristic_v1_medium": HEURISTIC_V1_MEDIUM,  # D3: 75% scaled
    "heuristic_v1_strong": HEURISTIC_V1_STRONG,  # D4: 100% balanced
    "heuristic_v1_expert": HEURISTIC_V1_EXPERT,  # D5: aggressive profile
    #
    # Canonical ladder-linked ids for pure heuristic play.
    # Each difficulty level now has genuinely different weights.
    "v1-heuristic-2": HEURISTIC_V1_WEAK,      # Weak: 50% scaled weights
    "v1-heuristic-3": HEURISTIC_V1_MEDIUM,    # Medium: 75% scaled weights
    "v1-heuristic-4": HEURISTIC_V1_STRONG,    # Strong: full balanced weights
    "v1-heuristic-5": HEURISTIC_V1_EXPERT,    # Expert: aggressive profile
    #
    # Higher difficulty levels (D6+) are typically MCTS/search-based,
    # not pure heuristic. When used as heuristic leaf evaluation,
    # they use the aggressive profile for maximum strength.
    "v1-heuristic-6": HEURISTIC_V1_AGGRESSIVE,
    "v1-heuristic-7": HEURISTIC_V1_AGGRESSIVE,
    "v1-heuristic-8": HEURISTIC_V1_AGGRESSIVE,
    "v1-heuristic-9": HEURISTIC_V1_AGGRESSIVE,
    "v1-heuristic-10": HEURISTIC_V1_AGGRESSIVE,
}


# --- Player-count-specific weight selection ---------------------------------

PLAYER_COUNT_PROFILE_MAP: dict[int, str] = {
    2: "heuristic_v1_2p",
    3: "heuristic_v1_3p",
    4: "heuristic_v1_4p",
}

# Board-type-specific profiles (keyed by board_type, num_players)
# This is the full 9-combination matrix. Each board×player combo gets
# its own CMA-ES optimized weights.
#
# Board type normalization: "square8" -> "sq8", "square19" -> "sq19", "hexagonal" -> "hex"
BOARD_ABBREV: dict[str, str] = {
    "square8": "sq8",
    "square19": "sq19",
    "hexagonal": "hex",
    "hex": "hex",  # Allow short form too
}

BOARD_PROFILE_MAP: dict[tuple, str] = {
    # Square8 (primary training board)
    ("square8", 2): "heuristic_v1_sq8_2p",
    ("square8", 3): "heuristic_v1_sq8_3p",
    ("square8", 4): "heuristic_v1_sq8_4p",
    # Square19 (larger board, different strategic dynamics)
    ("square19", 2): "heuristic_v1_sq19_2p",
    ("square19", 3): "heuristic_v1_sq19_3p",
    ("square19", 4): "heuristic_v1_sq19_4p",
    # Hexagonal (different topology)
    ("hexagonal", 2): "heuristic_v1_hex_2p",
    ("hexagonal", 3): "heuristic_v1_hex_3p",
    ("hexagonal", 4): "heuristic_v1_hex_4p",
}


def get_weights_for_player_count(
    num_players: int,
    fallback_profile: str = "heuristic_v1_balanced",
) -> HeuristicWeights:
    """Return the best weight profile for a given player count.

    This function enables automatic selection of player-count-specific weights
    that have been optimized through separate training runs. If no specific
    profile exists for the player count, falls back to the specified profile.

    Parameters
    ----------
    num_players:
        Number of players in the game (2, 3, or 4).
    fallback_profile:
        Profile ID to use if no player-specific profile exists.

    Returns
    -------
    HeuristicWeights
        The weight profile to use.

    Example
    -------
    >>> from app.ai.heuristic_weights import get_weights_for_player_count
    >>> weights = get_weights_for_player_count(3)  # Returns 3-player optimized
    """
    profile_id = PLAYER_COUNT_PROFILE_MAP.get(num_players)
    if profile_id and profile_id in HEURISTIC_WEIGHT_PROFILES:
        return HEURISTIC_WEIGHT_PROFILES[profile_id]
    return HEURISTIC_WEIGHT_PROFILES.get(fallback_profile, HEURISTIC_V1_BALANCED)


def get_weights_for_board(
    board_type: str,
    num_players: int,
    fallback_profile: str = "heuristic_v1_balanced",
) -> HeuristicWeights:
    """Return the best weight profile for a board type and player count.

    This function enables automatic selection of board-specific weights that
    have been optimized through separate training runs. Larger boards (e.g.,
    square19) have fundamentally different strategic requirements than smaller
    boards (e.g., square8).

    Selection priority:
    1. Board-type + player-count specific profile (e.g., square19 + 2p)
    2. Player-count specific profile (e.g., 3p profile for any board)
    3. Fallback profile

    Parameters
    ----------
    board_type:
        Board type identifier (e.g., "square8", "square19", "hexagonal").
    num_players:
        Number of players in the game (2, 3, or 4).
    fallback_profile:
        Profile ID to use if no specific profile exists.

    Returns
    -------
    HeuristicWeights
        The weight profile to use.

    Example
    -------
    >>> from app.ai.heuristic_weights import get_weights_for_board
    >>> weights = get_weights_for_board("square19", 2)  # Returns 19x19 2p optimized
    >>> weights = get_weights_for_board("square8", 3)   # Falls back to 3p profile
    """
    # First check for board-type + player-count specific profile
    board_profile_id = BOARD_PROFILE_MAP.get((board_type, num_players))
    if board_profile_id and board_profile_id in HEURISTIC_WEIGHT_PROFILES:
        return HEURISTIC_WEIGHT_PROFILES[board_profile_id]

    # Fall back to player-count specific profile
    player_profile_id = PLAYER_COUNT_PROFILE_MAP.get(num_players)
    if player_profile_id and player_profile_id in HEURISTIC_WEIGHT_PROFILES:
        return HEURISTIC_WEIGHT_PROFILES[player_profile_id]

    return HEURISTIC_WEIGHT_PROFILES.get(fallback_profile, HEURISTIC_V1_BALANCED)


def get_weights(profile_id: str) -> HeuristicWeights:
    """Return the weight profile for ``profile_id``.

    Callers should treat a missing id as "no override" and fall back to
    whatever defaults their caller provides (typically the balanced profile
    baked into :class:`HeuristicAI`). This helper deliberately does *not*
    raise if the profile is unknown to keep runtime behaviour backward
    compatible with earlier versions that did not know about personas.
    """

    return HEURISTIC_WEIGHT_PROFILES.get(profile_id, {})


# Mapping from difficulty level to weight profile for PURE heuristic AI
# (not used as leaf evaluation in search algorithms)
PURE_HEURISTIC_DIFFICULTY_PROFILES: dict[int, str] = {
    2: "heuristic_v1_weak",      # 50% scaled weights
    3: "heuristic_v1_medium",    # 75% scaled weights
    4: "heuristic_v1_strong",    # 100% balanced weights
    5: "heuristic_v1_expert",    # Aggressive profile
}


def get_pure_heuristic_profile(difficulty: int) -> str:
    """Get the weight profile ID for pure heuristic AI at a given difficulty.

    Pure heuristic AI (no search algorithm) uses difficulty-scaled weights
    where each difficulty level has genuinely different weights:
    - D2: Weak (50% scaled)
    - D3: Medium (75% scaled)
    - D4: Strong (100% balanced)
    - D5: Expert (aggressive profile)

    For difficulty >= 6, returns the aggressive profile since those levels
    typically use search algorithms with heuristic leaf evaluation.

    Parameters
    ----------
    difficulty:
        Difficulty level (2-10).

    Returns
    -------
    str
        Profile ID like "heuristic_v1_weak", "heuristic_v1_medium", etc.
    """
    if difficulty <= 2:
        return "heuristic_v1_weak"
    elif difficulty == 3:
        return "heuristic_v1_medium"
    elif difficulty == 4:
        return "heuristic_v1_strong"
    elif difficulty == 5:
        return "heuristic_v1_expert"
    else:
        # D6+ uses aggressive for maximum strength (typically with search)
        return "heuristic_v1_aggressive"


def get_pure_heuristic_weights(difficulty: int) -> HeuristicWeights:
    """Get the weight dict for pure heuristic AI at a given difficulty.

    Convenience wrapper around get_pure_heuristic_profile + get_weights.
    """
    profile_id = get_pure_heuristic_profile(difficulty)
    return HEURISTIC_WEIGHT_PROFILES.get(profile_id, HEURISTIC_V1_BALANCED)


def get_profile_key(board_type: str, num_players: int) -> str:
    """Get the canonical profile key for a board×player combination.

    This is the standard profile key format used by CMA-ES training scripts
    to save and load board-specific heuristic weights.

    Parameters
    ----------
    board_type:
        Board type identifier: "square8", "square19", "hexagonal", or "hex".
    num_players:
        Number of players (2, 3, or 4).

    Returns
    -------
    str
        Profile key in format "heuristic_v1_{board_abbrev}_{n}p".

    Example
    -------
    >>> get_profile_key("square8", 2)
    'heuristic_v1_sq8_2p'
    >>> get_profile_key("hexagonal", 3)
    'heuristic_v1_hex_3p'
    """
    board_abbrev = BOARD_ABBREV.get(board_type, board_type[:3])
    return f"heuristic_v1_{board_abbrev}_{num_players}p"


def get_legacy_profile_key(num_players: int) -> str:
    """Get the legacy player-count-only profile key (for backwards compat).

    DEPRECATED: Use get_profile_key(board_type, num_players) instead.

    Parameters
    ----------
    num_players:
        Number of players (2, 3, or 4).

    Returns
    -------
    str
        Legacy profile key in format "heuristic_v1_{n}p".
    """
    return f"heuristic_v1_{num_players}p"


TRAINED_PROFILES_ENV = "RINGRIFT_TRAINED_HEURISTIC_PROFILES"


def load_trained_profiles_if_available(
    path: str | None = None,
    *,
    mode: str = "override",
    suffix: str = "_trained",
) -> dict[str, HeuristicWeights]:
    """Load trained heuristic profiles from JSON and merge into the registry.

    Parameters
    ----------
    path:
        Optional explicit path to a JSON file produced by
        :mod:`app.training.train_heuristic_weights`. If omitted, the helper
        looks for the ``RINGRIFT_TRAINED_HEURISTIC_PROFILES`` environment
        variable.
    mode:
        - ``"override"``: replace existing entries in
          :data:`HEURISTIC_WEIGHT_PROFILES` with trained values (good for
          production toggles).
        - ``"suffix"``: register trained copies under new ids of the form
          ``f"{profile_id}{suffix}"`` while leaving the baseline ids
          untouched (good for A/B experiments).
    suffix:
        Suffix used when ``mode == "suffix"``.

    Returns
    -------
    Dict[str, HeuristicWeights]
        Mapping of *newly* registered profile ids to their weights.
    """

    if path is None:
        path = os.getenv(TRAINED_PROFILES_ENV)

    if not path or not os.path.exists(path):
        return {}

    alias_targets: list[tuple[str, str]] = []
    if mode == "override":
        # If callers provide only legacy player-count keys (heuristic_v1_2p/3p/4p),
        # propagate those trained weights into any board-specific ids that are
        # configured as strict aliases of the legacy keys. This avoids silently
        # ignoring trained weights when production ladder tiers reference
        # board×player ids such as heuristic_v1_sq8_3p.
        for (_board_type, num_players), board_profile_id in BOARD_PROFILE_MAP.items():
            legacy_profile_id = PLAYER_COUNT_PROFILE_MAP.get(num_players)
            if not legacy_profile_id:
                continue
            if (
                board_profile_id in HEURISTIC_WEIGHT_PROFILES
                and legacy_profile_id in HEURISTIC_WEIGHT_PROFILES
                and HEURISTIC_WEIGHT_PROFILES[board_profile_id]
                is HEURISTIC_WEIGHT_PROFILES[legacy_profile_id]
            ):
                alias_targets.append((board_profile_id, legacy_profile_id))

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    profiles = payload.get("profiles", {})
    loaded: dict[str, HeuristicWeights] = {}

    for pid, weights in profiles.items():
        if mode == "override":
            target_id = pid
        elif mode == "suffix":
            target_id = f"{pid}{suffix}"
        else:
            # Unknown mode: skip silently to preserve behaviour rather than
            # raising at runtime.
            continue

        HEURISTIC_WEIGHT_PROFILES[target_id] = dict(weights)
        loaded[target_id] = HEURISTIC_WEIGHT_PROFILES[target_id]

    if mode == "override" and alias_targets and isinstance(profiles, dict):
        for board_profile_id, legacy_profile_id in alias_targets:
            if legacy_profile_id not in loaded:
                continue
            # If the JSON payload included an explicit board-specific key,
            # prefer it over legacy-key propagation.
            if board_profile_id in profiles or board_profile_id in loaded:
                continue
            HEURISTIC_WEIGHT_PROFILES[board_profile_id] = dict(
                HEURISTIC_WEIGHT_PROFILES[legacy_profile_id]
            )
            loaded[board_profile_id] = HEURISTIC_WEIGHT_PROFILES[board_profile_id]

    return loaded
