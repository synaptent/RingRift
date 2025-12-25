#!/usr/bin/env python3
"""Diverse AI Configuration for High-Quality Training Data Generation.

This module provides a centralized configuration for diverse AI matchups
across all training-related scripts: soaks, tournaments, gauntlets, CMAES.

All 23 AI Types (Dec 2025):
- RANDOM: Pure random moves (weak baseline)
- HEURISTIC: Fast heuristic evaluation
- MINIMAX: Paranoid minimax search (assumes opponent collusion)
- GPU_MINIMAX: GPU-accelerated batched minimax
- MAXN: Max-N search (each player maximizes own score)
- BRS: Best-Reply Search (greedy best replies)
- MCTS: Monte Carlo Tree Search
- DESCENT: Gradient descent search
- NEURAL_DEMO: Experimental neural-only mode
- POLICY_ONLY: Direct NN policy without search
- GUMBEL_MCTS: Gumbel AlphaZero with Sequential Halving
- EBMO: Energy-Based Move Optimization
- GMO: Gradient Move Optimization (entropy-guided gradient ascent)
- GMO_V2: Enhanced GMO with attention and ensemble
- GMO_MCTS: GMO-guided MCTS tree search
- GMO_GUMBEL: GMO value network + Gumbel MCTS search
- IG_GMO: Information-Gain GMO (mutual information exploration)
- CAGE: Constraint-Aware Graph Energy-based optimization
- IMPROVED_MCTS: Advanced MCTS with PUCT + progressive widening
- HYBRID_NN: Fast heuristic + NN value ranking (hybrid)
- GNN: Graph Neural Network (message passing for hex geometry)
- HYBRID: CNN-GNN hybrid (CNN patterns + GNN connectivity)

GPU-Optimized Distribution:
- Prioritizes GUMBEL_MCTS (20%) and POLICY_ONLY (15%) for GPU utilization
- Includes diverse weak/strong matchups for robust training
- Includes all experimental AIs for maximum diversity
- Supports all board types

Usage:
    from app.training.diverse_ai_config import (
        ALL_AI_TYPES,
        GPU_OPTIMIZED_WEIGHTS,
        get_diverse_matchups,
        get_weighted_ai_type,
    )
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# All 23 AI types (Dec 2025)
ALL_AI_TYPES = [
    # Core production types
    "random",
    "heuristic",
    "minimax",
    "gpu_minimax",
    "maxn",
    "brs",
    "mcts",
    "descent",
    "neural_demo",
    "policy_only",
    "gumbel_mcts",
    # Experimental energy/gradient-based types
    "ebmo",
    "gmo",
    "gmo_v2",
    "gmo_mcts",
    "gmo_gumbel",
    "ig_gmo",
    "cage",
    # Advanced search types
    "improved_mcts",
    "hybrid_nn",
    # GNN-based types
    "gnn",
    "hybrid",
]

# GPU-heavy AI types (prioritized for GPU utilization)
GPU_AI_TYPES = [
    "gpu_minimax",
    "gumbel_mcts",
    "policy_only",
    "mcts",
    "descent",
    "gmo",
    "gmo_v2",
    "gmo_mcts",
    "gmo_gumbel",
    "improved_mcts",
    "hybrid_nn",
    "gnn",
    "hybrid",
]

# CPU AI types (can run without GPU)
CPU_AI_TYPES = [
    "random",
    "heuristic",
    "minimax",
    "maxn",
    "brs",
    "ebmo",
    "ig_gmo",
    "cage",
]

# Strong AI types (for asymmetric matchups)
STRONG_AI_TYPES = [
    "gumbel_mcts",
    "policy_only",
    "gpu_minimax",
    "mcts",
    "descent",
    "minimax",
    "maxn",
    "gmo_mcts",
    "gmo_gumbel",
    "improved_mcts",
    "hybrid_nn",
    "gnn",
    "hybrid",
]

# Experimental AI types (novel architectures for diversity)
EXPERIMENTAL_AI_TYPES = [
    "ebmo",
    "gmo",
    "gmo_v2",
    "gmo_mcts",
    "gmo_gumbel",
    "ig_gmo",
    "cage",
    "improved_mcts",
    "gnn",
    "hybrid",
]

# Weak AI types (for diversity and baseline)
WEAK_AI_TYPES = [
    "random",
    "heuristic",
    "brs",
]

# GPU-optimized weight distribution
# Prioritizes NN-guided types for high-quality training data
# Total must sum to 1.0
GPU_OPTIMIZED_WEIGHTS: dict[str, float] = {
    # Core high-quality search (44%)
    "gumbel_mcts": 0.18,    # 18% - Top priority, best quality search
    "policy_only": 0.12,    # 12% - Fast GPU policy (volume)
    "improved_mcts": 0.07,  # 7% - Advanced MCTS with PUCT
    "gmo_gumbel": 0.04,     # 4% - GMO + Gumbel hybrid
    "gmo_mcts": 0.03,       # 3% - GMO-guided MCTS
    # GNN-based (competitive with CNN - 12%)
    "gnn": 0.06,            # 6% - Graph Neural Network (competitive)
    "hybrid": 0.06,         # 6% - CNN-GNN hybrid (competitive)
    # Standard search (16%)
    "gpu_minimax": 0.04,    # 4% - GPU batched search
    "mcts": 0.04,           # 4% - MCTS exploration
    "descent": 0.04,        # 4% - Gradient search (NN-guided)
    "hybrid_nn": 0.04,      # 4% - Fast hybrid NN
    # Experimental (10%)
    "gmo": 0.03,            # 3% - Gradient Move Optimization
    "gmo_v2": 0.02,         # 2% - Enhanced GMO
    "ebmo": 0.02,           # 2% - Energy-based
    "ig_gmo": 0.01,         # 1% - Information-gain GMO
    "cage": 0.02,           # 2% - Constraint-aware
    # Baselines (18%)
    "minimax": 0.04,        # 4% - Paranoid search
    "maxn": 0.04,           # 4% - Max-N search
    "brs": 0.03,            # 3% - Fast best-reply
    "heuristic": 0.04,      # 4% - Baseline
    "neural_demo": 0.02,    # 2% - Experimental
    "random": 0.01,         # 1% - Weak diversity (minimal)
}

# CPU-optimized weight distribution (for CPU-only nodes)
# Total must sum to 1.0
CPU_OPTIMIZED_WEIGHTS: dict[str, float] = {
    # CPU-efficient search (60%)
    "minimax": 0.15,
    "maxn": 0.15,
    "mcts": 0.12,
    "descent": 0.10,
    "brs": 0.08,
    # Experimental CPU-friendly (18%)
    "ebmo": 0.05,
    "ig_gmo": 0.05,
    "cage": 0.04,
    "gmo": 0.04,            # Can run on CPU (slower)
    # Baselines (15%)
    "heuristic": 0.08,
    "random": 0.05,
    "neural_demo": 0.02,
    # GPU types (run on CPU with reduced weight - 7%)
    "gumbel_mcts": 0.03,
    "policy_only": 0.02,
    "gnn": 0.02,            # GNN can run on CPU
    # Skip GPU-only types
    "gpu_minimax": 0.00,
    "gmo_v2": 0.00,
    "gmo_mcts": 0.00,
    "gmo_gumbel": 0.00,
    "improved_mcts": 0.00,
    "hybrid_nn": 0.00,
    "hybrid": 0.00,
}

# Balanced distribution for tournaments
# Includes ALL AI types for maximum diversity in matchups
# Total must sum to 1.0
TOURNAMENT_WEIGHTS: dict[str, float] = {
    # Core search types (40%)
    "gumbel_mcts": 0.10,
    "policy_only": 0.08,
    "mcts": 0.07,
    "descent": 0.06,
    "improved_mcts": 0.05,
    "gmo_gumbel": 0.02,
    "gmo_mcts": 0.02,
    # GNN-based (competitive - 12%)
    "gnn": 0.06,
    "hybrid": 0.06,
    # Search variants (18%)
    "minimax": 0.05,
    "gpu_minimax": 0.05,
    "maxn": 0.05,
    "brs": 0.03,
    # Experimental types (20%)
    "gmo": 0.04,
    "gmo_v2": 0.03,
    "ebmo": 0.03,
    "ig_gmo": 0.02,
    "cage": 0.02,
    "hybrid_nn": 0.03,
    "neural_demo": 0.03,
    # Baselines (10%)
    "heuristic": 0.06,
    "random": 0.04,
}

# Robust training weights - emphasizes asymmetric matchups with random/heuristic
# This helps models learn correct value semantics by training against weak opponents
# where the value signal is clearer (NN should always beat random/heuristic)
# Total must sum to 1.0
ROBUST_TRAINING_WEIGHTS: dict[str, float] = {
    # High-quality search (35%)
    "gumbel_mcts": 0.10,
    "mcts": 0.07,
    "descent": 0.06,
    "policy_only": 0.06,
    "improved_mcts": 0.04,
    "gmo_gumbel": 0.02,
    # GNN-based (competitive results - 10%)
    "gnn": 0.05,
    "hybrid": 0.05,
    # Search variants (15%)
    "minimax": 0.05,
    "maxn": 0.04,
    "gpu_minimax": 0.04,
    "brs": 0.02,
    # Experimental (12%)
    "gmo": 0.03,
    "gmo_v2": 0.02,
    "gmo_mcts": 0.02,
    "ebmo": 0.02,
    "ig_gmo": 0.01,
    "cage": 0.01,
    "hybrid_nn": 0.01,
    # Weak baselines for asymmetric matchups (28%)
    "heuristic": 0.12,
    "random": 0.10,
    "neural_demo": 0.06,
}

# Supported board types including hex8
BOARD_TYPES = [
    "square8",
    "square19",
    "hexagonal",
    "hex8",  # Added hex8 support
]


@dataclass
class MatchupConfig:
    """Configuration for a specific AI matchup."""
    player1_type: str
    player2_type: str
    player3_type: str | None = None
    player4_type: str | None = None
    weight: float = 1.0
    description: str = ""

    @property
    def ai_types(self) -> list[str]:
        """Get list of AI types for all players."""
        types = [self.player1_type, self.player2_type]
        if self.player3_type:
            types.append(self.player3_type)
        if self.player4_type:
            types.append(self.player4_type)
        return types


@dataclass
class DiverseAIConfig:
    """Configuration for diverse AI training data generation."""
    board_type: str = "square8"
    num_players: int = 2
    use_gpu: bool = True
    weights: dict[str, float] = field(default_factory=lambda: GPU_OPTIMIZED_WEIGHTS.copy())
    include_asymmetric: bool = True  # Include strong vs weak matchups
    include_selfplay: bool = True    # Include same-AI matchups

    def get_weights(self) -> dict[str, float]:
        """Get appropriate weights based on GPU availability."""
        if self.use_gpu:
            return GPU_OPTIMIZED_WEIGHTS
        return CPU_OPTIMIZED_WEIGHTS


def get_weighted_ai_type(
    weights: dict[str, float] | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Select a random AI type based on weights.

    Args:
        weights: Weight distribution (default: GPU_OPTIMIZED_WEIGHTS)
        exclude: AI types to exclude from selection

    Returns:
        Selected AI type string
    """
    if weights is None:
        weights = GPU_OPTIMIZED_WEIGHTS

    if exclude:
        weights = {k: v for k, v in weights.items() if k not in exclude}

    # Normalize weights
    total = sum(weights.values())
    if total == 0:
        return random.choice(ALL_AI_TYPES)

    normalized = {k: v / total for k, v in weights.items()}

    # Weighted random selection
    r = random.random()
    cumulative = 0.0
    for ai_type, weight in normalized.items():
        cumulative += weight
        if r < cumulative:
            return ai_type

    return list(normalized.keys())[-1]


def get_diverse_matchups(
    num_players: int = 2,
    num_matchups: int = 20,
    config: DiverseAIConfig | None = None,
) -> list[MatchupConfig]:
    """Generate diverse AI matchup configurations.

    Args:
        num_players: Number of players (2-4)
        num_matchups: Number of unique matchups to generate
        config: Optional DiverseAIConfig for customization

    Returns:
        List of MatchupConfig objects
    """
    if config is None:
        config = DiverseAIConfig(num_players=num_players)

    weights = config.get_weights()
    matchups: list[MatchupConfig] = []
    seen: set = set()

    # Strong vs Strong matchups (high quality games)
    strong_matchups = int(num_matchups * 0.4)
    for _ in range(strong_matchups):
        p1 = get_weighted_ai_type(weights, exclude=WEAK_AI_TYPES)
        p2 = get_weighted_ai_type(weights, exclude=WEAK_AI_TYPES)

        key = tuple(sorted([p1, p2]))
        if key not in seen or len(seen) < num_matchups // 2:
            seen.add(key)

            if num_players == 2:
                matchups.append(MatchupConfig(
                    player1_type=p1,
                    player2_type=p2,
                    weight=weights.get(p1, 0.1) + weights.get(p2, 0.1),
                    description=f"{p1} vs {p2} (strong)",
                ))
            elif num_players == 3:
                p3 = get_weighted_ai_type(weights, exclude=WEAK_AI_TYPES)
                matchups.append(MatchupConfig(
                    player1_type=p1,
                    player2_type=p2,
                    player3_type=p3,
                    weight=weights.get(p1, 0.1) + weights.get(p2, 0.1) + weights.get(p3, 0.1),
                    description=f"{p1} vs {p2} vs {p3} (strong)",
                ))
            else:  # 4 players
                p3 = get_weighted_ai_type(weights, exclude=WEAK_AI_TYPES)
                p4 = get_weighted_ai_type(weights, exclude=WEAK_AI_TYPES)
                matchups.append(MatchupConfig(
                    player1_type=p1,
                    player2_type=p2,
                    player3_type=p3,
                    player4_type=p4,
                    weight=sum(weights.get(p, 0.1) for p in [p1, p2, p3, p4]),
                    description=f"{p1} vs {p2} vs {p3} vs {p4} (strong)",
                ))

    # Asymmetric matchups (strong vs weak for learning)
    if config.include_asymmetric:
        asymmetric_matchups = int(num_matchups * 0.3)
        for _ in range(asymmetric_matchups):
            strong = get_weighted_ai_type(weights, exclude=WEAK_AI_TYPES)
            weak = random.choice(WEAK_AI_TYPES)

            if num_players == 2:
                matchups.append(MatchupConfig(
                    player1_type=strong,
                    player2_type=weak,
                    weight=weights.get(strong, 0.1) * 1.5,  # Boost asymmetric
                    description=f"{strong} vs {weak} (asymmetric)",
                ))
            elif num_players >= 3:
                # Mix of strong and weak
                p3 = get_weighted_ai_type(weights)
                matchups.append(MatchupConfig(
                    player1_type=strong,
                    player2_type=weak,
                    player3_type=p3,
                    player4_type=get_weighted_ai_type(weights) if num_players == 4 else None,
                    weight=weights.get(strong, 0.1) * 1.2,
                    description=f"Asymmetric {num_players}p",
                ))

    # Self-play matchups (same AI type)
    if config.include_selfplay:
        selfplay_matchups = int(num_matchups * 0.2)
        for _ in range(selfplay_matchups):
            ai_type = get_weighted_ai_type(weights)
            if num_players == 2:
                matchups.append(MatchupConfig(
                    player1_type=ai_type,
                    player2_type=ai_type,
                    weight=weights.get(ai_type, 0.1) * 2,
                    description=f"{ai_type} self-play",
                ))
            else:
                matchups.append(MatchupConfig(
                    player1_type=ai_type,
                    player2_type=ai_type,
                    player3_type=ai_type,
                    player4_type=ai_type if num_players == 4 else None,
                    weight=weights.get(ai_type, 0.1) * 2,
                    description=f"{ai_type} {num_players}p self-play",
                ))

    # Random diverse matchups
    remaining = num_matchups - len(matchups)
    for _ in range(remaining):
        types = [get_weighted_ai_type(weights) for _ in range(num_players)]
        matchups.append(MatchupConfig(
            player1_type=types[0],
            player2_type=types[1],
            player3_type=types[2] if num_players >= 3 else None,
            player4_type=types[3] if num_players == 4 else None,
            weight=sum(weights.get(t, 0.1) for t in types),
            description=f"Diverse {num_players}p",
        ))

    return matchups


def get_ai_type_for_difficulty(difficulty: int, use_gpu: bool = True) -> str:
    """Map difficulty level to appropriate AI type.

    Args:
        difficulty: Difficulty level (1-10)
        use_gpu: Whether GPU is available

    Returns:
        Appropriate AI type string
    """
    if difficulty <= 2:
        return "random"
    elif difficulty <= 3:
        return "heuristic"
    elif difficulty <= 4:
        return "brs"
    elif difficulty <= 5:
        return "minimax" if not use_gpu else "gpu_minimax"
    elif difficulty <= 6:
        return "maxn"
    elif difficulty <= 7:
        return "mcts"
    elif difficulty <= 8:
        return "descent"
    elif difficulty <= 9:
        return "policy_only" if use_gpu else "descent"
    else:  # 10
        return "gumbel_mcts" if use_gpu else "mcts"


def get_training_ai_distribution(
    games_total: int,
    use_gpu: bool = True,
) -> dict[str, int]:
    """Get distribution of games per AI type for training.

    Args:
        games_total: Total number of games to generate
        use_gpu: Whether GPU is available

    Returns:
        Dict mapping AI type to number of games
    """
    weights = GPU_OPTIMIZED_WEIGHTS if use_gpu else CPU_OPTIMIZED_WEIGHTS

    distribution = {}
    remaining = games_total

    for ai_type, weight in sorted(weights.items(), key=lambda x: -x[1]):
        if weight > 0:
            count = int(games_total * weight)
            distribution[ai_type] = count
            remaining -= count

    # Distribute remaining games to top types
    if remaining > 0:
        top_types = sorted(weights.items(), key=lambda x: -x[1])[:3]
        for i, (ai_type, _) in enumerate(top_types):
            if i < remaining:
                distribution[ai_type] = distribution.get(ai_type, 0) + 1

    return distribution


# Convenience constants for common configurations
SELFPLAY_CONFIG_GPU = DiverseAIConfig(use_gpu=True)
SELFPLAY_CONFIG_CPU = DiverseAIConfig(use_gpu=False)
TOURNAMENT_CONFIG = DiverseAIConfig(
    use_gpu=True,
    weights=TOURNAMENT_WEIGHTS,
    include_asymmetric=False,  # Tournaments should be fair matchups
)
