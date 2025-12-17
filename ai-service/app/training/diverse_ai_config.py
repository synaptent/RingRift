#!/usr/bin/env python3
"""Diverse AI Configuration for High-Quality Training Data Generation.

This module provides a centralized configuration for diverse AI matchups
across all training-related scripts: soaks, tournaments, gauntlets, CMAES.

All 11 AI Types:
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

GPU-Optimized Distribution:
- Prioritizes GUMBEL_MCTS (20%) and POLICY_ONLY (15%) for GPU utilization
- Includes diverse weak/strong matchups for robust training
- Supports hex8 board type

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
from enum import Enum
from typing import Dict, List, Optional, Tuple

# All 11 AI types
ALL_AI_TYPES = [
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
]

# GPU-heavy AI types (prioritized for GPU utilization)
GPU_AI_TYPES = [
    "gpu_minimax",
    "gumbel_mcts",
    "policy_only",
    "mcts",
    "descent",
]

# CPU AI types
CPU_AI_TYPES = [
    "random",
    "heuristic",
    "minimax",
    "maxn",
    "brs",
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
]

# Weak AI types (for diversity and baseline)
WEAK_AI_TYPES = [
    "random",
    "heuristic",
    "brs",
]

# GPU-optimized weight distribution
# Prioritizes GPU-heavy types for 36-44% GPU utilization
GPU_OPTIMIZED_WEIGHTS: Dict[str, float] = {
    "gumbel_mcts": 0.20,    # 20% - Top priority, best quality
    "policy_only": 0.15,    # 15% - Fast GPU policy
    "gpu_minimax": 0.12,    # 12% - GPU batched search
    "mcts": 0.10,           # 10% - MCTS exploration
    "descent": 0.10,        # 10% - Gradient search
    "minimax": 0.08,        # 8% - Paranoid search
    "maxn": 0.08,           # 8% - Max-N search
    "brs": 0.07,            # 7% - Fast best-reply
    "heuristic": 0.05,      # 5% - Baseline
    "random": 0.03,         # 3% - Weak diversity
    "neural_demo": 0.02,    # 2% - Experimental
}

# CPU-optimized weight distribution (for CPU-only nodes)
CPU_OPTIMIZED_WEIGHTS: Dict[str, float] = {
    "minimax": 0.20,
    "maxn": 0.18,
    "mcts": 0.15,
    "descent": 0.12,
    "brs": 0.12,
    "heuristic": 0.10,
    "random": 0.08,
    "gumbel_mcts": 0.03,    # Can still run on CPU
    "policy_only": 0.02,
    "gpu_minimax": 0.00,    # Skip GPU types on CPU
    "neural_demo": 0.00,
}

# Balanced distribution for tournaments
TOURNAMENT_WEIGHTS: Dict[str, float] = {
    "gumbel_mcts": 0.15,
    "policy_only": 0.12,
    "mcts": 0.12,
    "descent": 0.12,
    "minimax": 0.10,
    "gpu_minimax": 0.10,
    "maxn": 0.10,
    "brs": 0.08,
    "heuristic": 0.06,
    "random": 0.03,
    "neural_demo": 0.02,
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
    player3_type: Optional[str] = None
    player4_type: Optional[str] = None
    weight: float = 1.0
    description: str = ""

    @property
    def ai_types(self) -> List[str]:
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
    weights: Dict[str, float] = field(default_factory=lambda: GPU_OPTIMIZED_WEIGHTS.copy())
    include_asymmetric: bool = True  # Include strong vs weak matchups
    include_selfplay: bool = True    # Include same-AI matchups

    def get_weights(self) -> Dict[str, float]:
        """Get appropriate weights based on GPU availability."""
        if self.use_gpu:
            return GPU_OPTIMIZED_WEIGHTS
        return CPU_OPTIMIZED_WEIGHTS


def get_weighted_ai_type(
    weights: Optional[Dict[str, float]] = None,
    exclude: Optional[List[str]] = None,
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
    config: Optional[DiverseAIConfig] = None,
) -> List[MatchupConfig]:
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
    matchups: List[MatchupConfig] = []
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
) -> Dict[str, int]:
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
