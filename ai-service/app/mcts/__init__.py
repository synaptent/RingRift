"""MCTS (Monte Carlo Tree Search) implementations for RingRift.

This package provides advanced MCTS algorithms with features like:
- PUCT exploration (AlphaZero-style)
- Progressive widening for large branching factors
- Virtual loss for parallel search
- Transposition tables for position caching
- Tree reuse between moves

Usage:
    from app.mcts import ImprovedMCTS, MCTSConfig, ParallelMCTS

    # For use as an AI player, use the integrated AI class:
    from app.ai.improved_mcts_ai import ImprovedMCTSAI
"""

from app.mcts.improved_mcts import (
    GameState,
    ImprovedMCTS,
    MCTSConfig,
    MCTSNode,
    MCTSWithPonder,
    NeuralNetworkInterface,
    ParallelMCTS,
    TranspositionTable,
)

__all__ = [
    "ImprovedMCTS",
    "MCTSConfig",
    "MCTSNode",
    "MCTSWithPonder",
    "NeuralNetworkInterface",
    "ParallelMCTS",
    "TranspositionTable",
    "GameState",
]
