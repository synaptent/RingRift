"""Selfplay Job Configurations - Extracted from P2POrchestrator._manage_cluster_jobs().

January 2026: Extracts selfplay config data to reduce p2p_orchestrator.py size.

This module provides:
- SELFPLAY_CONFIGS: Priority-weighted selfplay job configurations
- get_filtered_configs(): Filter configs based on node memory
- get_weighted_configs(): Expand configs by priority for weighted selection
- get_unique_configs(): Get unique (board_type, num_players) combinations

Priority Tiers:
- 10-12: GUMBEL MCTS (highest priority, GPU-accelerated quality data)
- 8: Mixed mode (varied AI matchups for diversity)
- 7: Underrepresented combinations (nn-only, nn-vs-mcts)
- 6: Cross-AI matches (heuristic/random vs strong AI)
- 5: Neural network modes and asymmetric tournaments
- 4: Tournament-varied (max diversity)
- 3: CPU-bound methods (MCTS, Descent for CPU instances)
- 2: Minimax (classical approach, variety)

Usage:
    from scripts.p2p.config.selfplay_job_configs import (
        SELFPLAY_CONFIGS,
        get_filtered_configs,
        get_weighted_configs,
        get_unique_configs,
    )

    # Filter for low-memory nodes
    configs = get_filtered_configs(node_memory_gb=32)

    # Get weighted list for random selection
    weighted = get_weighted_configs(configs)

    # Get unique board/player combinations
    unique = get_unique_configs(configs)
"""

from __future__ import annotations

from typing import Any

# ============================================================================
# Priority Constants
# ============================================================================

PRIORITY_GUMBEL_HIGH = 12      # Underrepresented Gumbel configs (3p/4p)
PRIORITY_GUMBEL_STANDARD = 10  # Standard Gumbel configs (2p)
PRIORITY_MIXED = 8             # Mixed mode for AI diversity
PRIORITY_UNDERREP = 7          # Underrepresented specific modes
PRIORITY_CROSS_AI = 6          # Cross-AI matches
PRIORITY_NN = 5                # Neural network modes
PRIORITY_TOURNAMENT = 4        # Tournament-varied
PRIORITY_CPU = 3               # CPU-bound methods
PRIORITY_MINIMAX = 2           # Classical minimax


# ============================================================================
# Selfplay Configuration Data
# ============================================================================

SELFPLAY_CONFIGS: list[dict[str, Any]] = [
    # ========================================================================
    # GUMBEL MCTS - HIGHEST PRIORITY (70% of jobs should use Gumbel)
    # GPU-accelerated Gumbel Top-K MCTS for high-quality training data
    # 3p/4p get priority 12 (underrepresented), 2p get priority 10
    # ========================================================================
    {"board_type": "hex8", "num_players": 2, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_STANDARD},
    {"board_type": "hex8", "num_players": 3, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "hex8", "num_players": 4, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "square8", "num_players": 2, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_STANDARD},
    {"board_type": "square8", "num_players": 3, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "square8", "num_players": 4, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_STANDARD},
    {"board_type": "square19", "num_players": 2, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "square19", "num_players": 3, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "square19", "num_players": 4, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},
    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "gumbel-mcts", "priority": PRIORITY_GUMBEL_HIGH},

    # ========================================================================
    # UNDERREPRESENTED COMBINATIONS - MIXED MODE (PRIORITY 8)
    # "mixed" mode provides varied AI matchups (NNUE, MCTS, heuristic combos)
    # for maximum training data diversity (~30% of jobs)
    # ========================================================================
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "hex8", "num_players": 2, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "hex8", "num_players": 3, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "hex8", "num_players": 4, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "square19", "num_players": 3, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "square19", "num_players": 2, "engine_mode": "mixed", "priority": PRIORITY_MIXED},
    {"board_type": "square19", "num_players": 4, "engine_mode": "mixed", "priority": PRIORITY_MIXED},

    # ========================================================================
    # UNDERREPRESENTED COMBINATIONS (PRIORITY 7)
    # Specific AI matchup modes for variety
    # ========================================================================
    {"board_type": "square19", "num_players": 2, "engine_mode": "nn-only", "priority": PRIORITY_UNDERREP},
    {"board_type": "square19", "num_players": 2, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_UNDERREP},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "nn-only", "priority": PRIORITY_UNDERREP},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_UNDERREP},
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "nn-only", "priority": PRIORITY_UNDERREP},
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_UNDERREP},

    # ========================================================================
    # SQUARE8 MULTI-PLAYER WITH MIXED MODE (PRIORITY 7/6)
    # ========================================================================
    {"board_type": "square8", "num_players": 3, "engine_mode": "mixed", "priority": PRIORITY_UNDERREP},
    {"board_type": "square8", "num_players": 4, "engine_mode": "mixed", "priority": PRIORITY_UNDERREP},
    {"board_type": "square8", "num_players": 2, "engine_mode": "mixed", "priority": PRIORITY_CROSS_AI},

    # ========================================================================
    # CROSS-AI MATCHES (PRIORITY 6) - Variety via asymmetric opponents
    # heuristic/random vs strong AI (MCTS, Minimax, Descent, NN)
    # ========================================================================
    {"board_type": "square8", "num_players": 2, "engine_mode": "heuristic-vs-nn", "priority": PRIORITY_CROSS_AI},
    {"board_type": "square8", "num_players": 2, "engine_mode": "heuristic-vs-mcts", "priority": PRIORITY_CROSS_AI},
    {"board_type": "square8", "num_players": 2, "engine_mode": "random-vs-mcts", "priority": PRIORITY_CROSS_AI},
    {"board_type": "square8", "num_players": 3, "engine_mode": "heuristic-vs-nn", "priority": PRIORITY_CROSS_AI},
    {"board_type": "square8", "num_players": 3, "engine_mode": "heuristic-vs-mcts", "priority": PRIORITY_CROSS_AI},
    {"board_type": "square19", "num_players": 2, "engine_mode": "heuristic-vs-mcts", "priority": PRIORITY_CROSS_AI},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "heuristic-vs-mcts", "priority": PRIORITY_CROSS_AI},
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "heuristic-vs-mcts", "priority": PRIORITY_CROSS_AI},

    # ========================================================================
    # NEURAL NETWORK MODES (PRIORITY 5)
    # ========================================================================
    {"board_type": "square8", "num_players": 2, "engine_mode": "nn-only", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 3, "engine_mode": "nn-only", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 4, "engine_mode": "nn-only", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 2, "engine_mode": "best-vs-pool", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 3, "engine_mode": "best-vs-pool", "priority": PRIORITY_NN},
    {"board_type": "square19", "num_players": 3, "engine_mode": "nn-only", "priority": PRIORITY_NN},
    {"board_type": "square19", "num_players": 4, "engine_mode": "nn-only", "priority": PRIORITY_NN},
    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "nn-only", "priority": PRIORITY_NN},

    # ========================================================================
    # ASYMMETRIC TOURNAMENT MODES (PRIORITY 5) - NN vs other AI types
    # ========================================================================
    {"board_type": "square8", "num_players": 2, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 3, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 4, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 2, "engine_mode": "nn-vs-minimax", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 3, "engine_mode": "nn-vs-minimax", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 2, "engine_mode": "nn-vs-descent", "priority": PRIORITY_NN},
    {"board_type": "square8", "num_players": 3, "engine_mode": "nn-vs-descent", "priority": PRIORITY_NN},
    {"board_type": "square19", "num_players": 3, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_NN},
    {"board_type": "square19", "num_players": 4, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_NN},
    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "nn-vs-mcts", "priority": PRIORITY_NN},

    # ========================================================================
    # TOURNAMENT-VARIED (PRIORITY 4) - Max diversity, always includes NN
    # ========================================================================
    {"board_type": "square8", "num_players": 2, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},
    {"board_type": "square8", "num_players": 3, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},
    {"board_type": "square8", "num_players": 4, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},
    {"board_type": "square19", "num_players": 2, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},
    {"board_type": "square19", "num_players": 3, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "tournament-varied", "priority": PRIORITY_TOURNAMENT},

    # ========================================================================
    # CPU-BOUND AI METHODS (PRIORITY 3) - MCTS, Descent, Minimax
    # For CPU-only instances and variety
    # ========================================================================
    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "mcts-only", "priority": PRIORITY_CPU},
    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "descent-only", "priority": PRIORITY_CPU},
    {"board_type": "square19", "num_players": 4, "engine_mode": "mcts-only", "priority": PRIORITY_CPU},
    {"board_type": "square19", "num_players": 3, "engine_mode": "mcts-only", "priority": PRIORITY_CPU},
    {"board_type": "square19", "num_players": 3, "engine_mode": "descent-only", "priority": PRIORITY_CPU},
    {"board_type": "square8", "num_players": 2, "engine_mode": "mcts-only", "priority": PRIORITY_CPU},
    {"board_type": "square8", "num_players": 2, "engine_mode": "descent-only", "priority": PRIORITY_CPU},
    {"board_type": "square8", "num_players": 3, "engine_mode": "descent-only", "priority": PRIORITY_CPU},
    {"board_type": "square8", "num_players": 4, "engine_mode": "mcts-only", "priority": PRIORITY_CPU},

    # ========================================================================
    # MINIMAX (PRIORITY 2) - Classical approach, good for variety
    # ========================================================================
    {"board_type": "square8", "num_players": 2, "engine_mode": "minimax-only", "priority": PRIORITY_MINIMAX},
    {"board_type": "square8", "num_players": 3, "engine_mode": "minimax-only", "priority": PRIORITY_MINIMAX},
    {"board_type": "square19", "num_players": 2, "engine_mode": "descent-only", "priority": PRIORITY_MINIMAX},
    {"board_type": "square19", "num_players": 2, "engine_mode": "minimax-only", "priority": PRIORITY_MINIMAX},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "descent-only", "priority": PRIORITY_MINIMAX},
    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "minimax-only", "priority": PRIORITY_MINIMAX},

    # NO PURE HEURISTIC-ONLY MODES - all modes include at least one
    # strong AI (NN/MCTS/Descent/Minimax) for quality training data
]


# ============================================================================
# Memory Threshold for Board Filtering
# ============================================================================

LOW_MEMORY_THRESHOLD_GB = 48  # Nodes below this get square8 only


# ============================================================================
# Helper Functions
# ============================================================================

def get_filtered_configs(
    node_memory_gb: int | None = None,
    board_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter selfplay configs based on node constraints.

    Args:
        node_memory_gb: Node's available memory in GB. If < 48GB,
            filters to square8 only to prevent OOM.
        board_types: Optional list of allowed board types.

    Returns:
        Filtered list of selfplay configurations.
    """
    configs = SELFPLAY_CONFIGS

    # Low memory nodes should stick to square8 to avoid OOM
    if node_memory_gb and node_memory_gb < LOW_MEMORY_THRESHOLD_GB:
        configs = [cfg for cfg in configs if cfg.get("board_type") == "square8"]

    # Additional board type filtering if specified
    if board_types:
        configs = [cfg for cfg in configs if cfg.get("board_type") in board_types]

    return configs


def get_weighted_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand configs by priority for weighted random selection.

    Higher priority configs appear more times in the list, giving them
    proportionally higher chances of being selected.

    Args:
        configs: List of selfplay configurations with 'priority' keys.

    Returns:
        Expanded list where each config appears 'priority' times.
    """
    weighted = []
    for cfg in configs:
        priority = cfg.get("priority", 1)
        weighted.extend([cfg] * priority)
    return weighted


def get_unique_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Get unique (board_type, num_players) combinations.

    Useful for round-robin distribution to ensure all configs get coverage.

    Args:
        configs: List of selfplay configurations.

    Returns:
        List with one config per unique (board_type, num_players) pair.
    """
    seen = {}
    for cfg in configs:
        key = (cfg["board_type"], cfg["num_players"])
        if key not in seen:
            seen[key] = cfg
    return list(seen.values())


def get_configs_for_engine_mode(engine_mode: str) -> list[dict[str, Any]]:
    """Get all configs that use a specific engine mode.

    Args:
        engine_mode: Engine mode to filter by (e.g., "gumbel-mcts").

    Returns:
        List of configs using that engine mode.
    """
    return [cfg for cfg in SELFPLAY_CONFIGS if cfg.get("engine_mode") == engine_mode]


def get_gumbel_configs() -> list[dict[str, Any]]:
    """Get all Gumbel MCTS configurations (highest quality).

    Returns:
        List of Gumbel MCTS configs.
    """
    return get_configs_for_engine_mode("gumbel-mcts")


# ============================================================================
# DIVERSE_PROFILES - Weighted profiles for _auto_start_selfplay
# ============================================================================
# These profiles are used for weighted random sampling on idle GPU nodes.
# Each profile has a 'weight' for selection probability and targets different
# aspects of game understanding for high-quality training data.

DIVERSE_PROFILES: list[dict[str, Any]] = [
    # High-quality neural-guided profiles (50% of games)
    {
        "engine_mode": "gumbel-mcts",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.18,
        "description": "Gumbel MCTS 2P hex8 - highest quality",
    },
    {
        "engine_mode": "policy-only",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.12,
        "description": "Policy-only 2P hex8 - fast NN inference",
    },
    {
        "engine_mode": "nnue-guided",
        "board_type": "square8",
        "num_players": 2,
        "profile": "aggressive",
        "weight": 0.08,
        "description": "NNUE-guided 2P square - aggressive style",
    },
    {
        "engine_mode": "gumbel-mcts",
        "board_type": "square8",
        "num_players": 3,
        "profile": "balanced",
        "weight": 0.06,
        "description": "Gumbel MCTS 3P square - multiplayer strategy",
    },
    {
        "engine_mode": "mcts",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "territorial",
        "weight": 0.06,
        "description": "MCTS 2P hex8 - territorial focus",
    },
    # MaxN/BRS multiplayer profiles (15% of games)
    # Benchmarks show: MaxN >> Descent in 3P/4P, MaxN ≈ BRS
    {
        "engine_mode": "maxn",
        "board_type": "hex8",
        "num_players": 3,
        "profile": "balanced",
        "weight": 0.05,
        "description": "MaxN 3P hex8 - optimal multiplayer search",
    },
    {
        "engine_mode": "maxn",
        "board_type": "square8",
        "num_players": 4,
        "profile": "balanced",
        "weight": 0.04,
        "description": "MaxN 4P square - best for 4-player",
    },
    {
        "engine_mode": "brs",
        "board_type": "hex8",
        "num_players": 3,
        "profile": "aggressive",
        "weight": 0.03,
        "description": "BRS 3P hex8 - fast multiplayer search",
    },
    {
        "engine_mode": "brs",
        "board_type": "square8",
        "num_players": 4,
        "profile": "territorial",
        "weight": 0.03,
        "description": "BRS 4P square - territorial multiplayer",
    },
    # GPU-accelerated throughput profiles (25% of games)
    {
        "engine_mode": "heuristic-only",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.10,
        "description": "GPU heuristic 2P hex8 - fast throughput",
    },
    {
        "engine_mode": "heuristic-only",
        "board_type": "square8",
        "num_players": 2,
        "profile": "defensive",
        "weight": 0.07,
        "description": "GPU heuristic 2P square - defensive style",
    },
    {
        "engine_mode": "heuristic-only",
        "board_type": "hex8",
        "num_players": 4,
        "profile": "balanced",
        "weight": 0.05,
        "description": "GPU heuristic 4P hex8 - large multiplayer",
    },
    # Exploration profiles (10% of games)
    {
        "engine_mode": "mixed",
        "board_type": "square19",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.04,
        "description": "Mixed 2P large board - strategic depth",
    },
    {
        "engine_mode": "nnue-guided",
        "board_type": "hex8",
        "num_players": 3,
        "profile": "aggressive",
        "weight": 0.04,
        "description": "NNUE 3P hex8 - aggressive multiplayer",
    },
    {
        "engine_mode": "policy-only",
        "board_type": "square8",
        "num_players": 4,
        "profile": "territorial",
        "weight": 0.05,
        "description": "Policy 4P square - territory control",
    },
    # Large board profiles (square19, hexagonal) - lighter engines for feasible throughput
    {
        "engine_mode": "heuristic-only",
        "board_type": "square19",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.03,
        "description": "Heuristic 2P square19 - fast large board",
    },
    {
        "engine_mode": "heuristic-only",
        "board_type": "hexagonal",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.03,
        "description": "Heuristic 2P hexagonal - fast large board",
    },
    {
        "engine_mode": "brs",
        "board_type": "square19",
        "num_players": 3,
        "profile": "balanced",
        "weight": 0.02,
        "description": "BRS 3P square19 - multiplayer large board",
    },
    {
        "engine_mode": "brs",
        "board_type": "hexagonal",
        "num_players": 3,
        "profile": "balanced",
        "weight": 0.02,
        "description": "BRS 3P hexagonal - multiplayer large board",
    },
    {
        "engine_mode": "maxn",
        "board_type": "square19",
        "num_players": 4,
        "profile": "balanced",
        "weight": 0.02,
        "description": "MaxN 4P square19 - high quality 4-player",
    },
    {
        "engine_mode": "maxn",
        "board_type": "hexagonal",
        "num_players": 4,
        "profile": "balanced",
        "weight": 0.02,
        "description": "MaxN 4P hexagonal - high quality 4-player",
    },
    # Full diversity profiles - Minimax (2P paranoid search), Descent (stochastic), Random (baseline)
    {
        "engine_mode": "nn-minimax",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.02,
        "description": "NN-Minimax 2P hex8 - deep alpha-beta search",
    },
    {
        "engine_mode": "nn-minimax",
        "board_type": "square8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.02,
        "description": "NN-Minimax 2P square - tactical search",
    },
    {
        "engine_mode": "nn-descent",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.01,
        "description": "NN-Descent 2P hex8 - stochastic exploration",
    },
    {
        "engine_mode": "nn-descent",
        "board_type": "square8",
        "num_players": 3,
        "profile": "balanced",
        "weight": 0.01,
        "description": "NN-Descent 3P square - multiplayer descent",
    },
    {
        "engine_mode": "random",
        "board_type": "hex8",
        "num_players": 2,
        "profile": "balanced",
        "weight": 0.005,
        "description": "Random 2P hex8 - baseline diversity",
    },
    {
        "engine_mode": "random",
        "board_type": "square8",
        "num_players": 4,
        "profile": "balanced",
        "weight": 0.005,
        "description": "Random 4P square - multiplayer baseline",
    },
]


def get_diverse_profile_weights() -> list[float]:
    """Get weights from DIVERSE_PROFILES for random.choices().

    Returns:
        List of weights corresponding to each profile.
    """
    return [p["weight"] for p in DIVERSE_PROFILES]


def select_diverse_profiles(k: int = 1) -> list[dict[str, Any]]:
    """Select k profiles using weighted random sampling.

    Args:
        k: Number of profiles to select.

    Returns:
        List of selected profile dicts.
    """
    import random
    weights = get_diverse_profile_weights()
    return random.choices(DIVERSE_PROFILES, weights=weights, k=k)
