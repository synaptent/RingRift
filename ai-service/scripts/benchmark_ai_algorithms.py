#!/usr/bin/env python3
"""Benchmark tournament for experimental AI algorithms.

Compares GMO, EBMO, IG-GMO, GMO-MCTS, Improved MCTS against baselines
across different board types to identify algorithms with highest Elo potential.

Usage:
    # Quick benchmark (10 games per matchup)
    python scripts/benchmark_ai_algorithms.py --board square8 --num-players 2 --games 10

    # Full benchmark (50 games, all algorithms)
    python scripts/benchmark_ai_algorithms.py --board square8 --num-players 2 --games 50 --full

    # Cluster execution with GPU
    python scripts/benchmark_ai_algorithms.py --board square8 --device cuda --games 100

Results are saved to data/benchmarks/ with Elo ratings and win rates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.base import BaseAI
from app.ai.factory import AIFactory
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType, GameState, GameStatus
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single match."""
    player1_agent: str
    player2_agent: str
    winner: int  # 1, 2, or 0 for draw
    num_moves: int
    duration_ms: float
    game_seed: int


@dataclass
class AgentStats:
    """Statistics for an agent."""
    agent_id: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_moves: int = 0
    games_played: int = 0
    elo: float = 1500.0

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tournament."""
    board_type: str = "square8"
    num_players: int = 2
    games_per_matchup: int = 20
    device: str = "cpu"
    nn_model_id: str | None = None
    think_time_ms: int = 500  # Reduced from 2000ms for faster benchmark
    base_seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("data/benchmarks"))

    # Algorithm selection
    include_gmo: bool = True
    include_ebmo: bool = True
    include_ig_gmo: bool = True
    include_gmo_mcts: bool = True
    include_improved_mcts: bool = True
    include_gumbel: bool = True
    include_policy_only: bool = True
    include_mcts_baseline: bool = True


# Elo calculation constants
K_FACTOR = 32
INITIAL_ELO = 1500.0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (elo_b - elo_a) / 400))


def update_elo(
    elo_a: float,
    elo_b: float,
    score_a: float,  # 1 for win, 0.5 for draw, 0 for loss
) -> tuple[float, float]:
    """Update Elo ratings based on match result."""
    expected_a = expected_score(elo_a, elo_b)
    expected_b = 1.0 - expected_a

    new_elo_a = elo_a + K_FACTOR * (score_a - expected_a)
    new_elo_b = elo_b + K_FACTOR * ((1 - score_a) - expected_b)

    return new_elo_a, new_elo_b


def create_agent(
    agent_id: str,
    player_number: int,
    config: BenchmarkConfig,
    game_seed: int,
) -> BaseAI:
    """Create an AI agent for the benchmark."""
    rng_seed = (game_seed * 104729 + player_number * 7919) & 0xFFFFFFFF

    ai_config = AIConfig(
        difficulty=7,
        think_time=config.think_time_ms,
        randomness=0.05,
        rng_seed=rng_seed,
        nn_model_id=config.nn_model_id,
        use_neural_net=config.nn_model_id is not None,
    )

    return AIFactory.create_for_tournament(
        agent_id=agent_id,
        player_number=player_number,
        board_type=config.board_type,
        num_players=config.num_players,
        rng_seed=rng_seed,
        nn_model_id=config.nn_model_id,
    )


def play_match(
    agent1_id: str,
    agent2_id: str,
    config: BenchmarkConfig,
    game_seed: int,
) -> MatchResult:
    """Play a single match between two agents."""
    start_time = time.time()

    # Create agents
    agent1 = create_agent(agent1_id, 1, config, game_seed)
    agent2 = create_agent(agent2_id, 2, config, game_seed)
    agents = {1: agent1, 2: agent2}

    # Initialize game using create_initial_state
    board_type = BoardType(config.board_type)
    game_state = create_initial_state(board_type, config.num_players)

    num_moves = 0
    max_moves = 500  # Prevent infinite games

    def is_game_over(state: GameState) -> bool:
        return state.game_status != GameStatus.ACTIVE

    while not is_game_over(game_state) and num_moves < max_moves:
        current_player = game_state.current_player
        ai = agents[current_player]

        move = ai.select_move(game_state)
        if move is None:
            break

        game_state = GameEngine.apply_move(game_state, move)
        num_moves += 1

    # Determine winner
    if is_game_over(game_state) and game_state.winner:
        winner = game_state.winner
    else:
        winner = 0  # Draw or timeout

    duration_ms = (time.time() - start_time) * 1000

    return MatchResult(
        player1_agent=agent1_id,
        player2_agent=agent2_id,
        winner=winner,
        num_moves=num_moves,
        duration_ms=duration_ms,
        game_seed=game_seed,
    )


def run_tournament(
    agent_ids: list[str],
    config: BenchmarkConfig,
) -> dict[str, AgentStats]:
    """Run round-robin tournament between all agents."""
    stats: dict[str, AgentStats] = {
        agent_id: AgentStats(agent_id=agent_id)
        for agent_id in agent_ids
    }

    # Generate all matchups (round-robin)
    matchups = []
    for i, agent1 in enumerate(agent_ids):
        for agent2 in agent_ids[i + 1:]:
            # Play both sides
            for _ in range(config.games_per_matchup // 2):
                matchups.append((agent1, agent2))
                matchups.append((agent2, agent1))

    # Shuffle to reduce ordering bias
    random.seed(config.base_seed)
    random.shuffle(matchups)

    logger.info(f"Running {len(matchups)} games across {len(agent_ids)} agents")

    for idx, (agent1_id, agent2_id) in enumerate(matchups):
        game_seed = config.base_seed + idx * 1_000_003

        try:
            result = play_match(agent1_id, agent2_id, config, game_seed)

            # Update stats
            stats[agent1_id].games_played += 1
            stats[agent2_id].games_played += 1
            stats[agent1_id].total_moves += result.num_moves
            stats[agent2_id].total_moves += result.num_moves

            if result.winner == 1:
                stats[agent1_id].wins += 1
                stats[agent2_id].losses += 1
                score_1 = 1.0
            elif result.winner == 2:
                stats[agent1_id].losses += 1
                stats[agent2_id].wins += 1
                score_1 = 0.0
            else:
                stats[agent1_id].draws += 1
                stats[agent2_id].draws += 1
                score_1 = 0.5

            # Update Elo
            new_elo_1, new_elo_2 = update_elo(
                stats[agent1_id].elo,
                stats[agent2_id].elo,
                score_1,
            )
            stats[agent1_id].elo = new_elo_1
            stats[agent2_id].elo = new_elo_2

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{len(matchups)} games "
                    f"({agent1_id} vs {agent2_id}: {'P1' if result.winner == 1 else 'P2' if result.winner == 2 else 'draw'})"
                )

        except Exception as e:
            logger.error(f"Error in match {agent1_id} vs {agent2_id}: {e}")
            continue

    return stats


def get_agent_list(config: BenchmarkConfig) -> list[str]:
    """Get list of agents to include in benchmark."""
    agents = []

    # Experimental algorithms (Priority 2 targets)
    if config.include_gmo:
        agents.append("gmo")
    if config.include_ebmo:
        agents.append("ebmo")
    if config.include_ig_gmo:
        agents.append("ig_gmo")
    if config.include_gmo_mcts:
        agents.append("gmo_mcts")
    if config.include_improved_mcts:
        agents.append("improved_mcts")

    # Strong baselines
    if config.include_gumbel:
        agents.append("gumbel_mcts")
    if config.include_policy_only:
        agents.append("policy_only")
    if config.include_mcts_baseline:
        agents.append("mcts_500")

    return agents


def save_results(
    stats: dict[str, AgentStats],
    config: BenchmarkConfig,
) -> Path:
    """Save benchmark results to JSON."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{config.board_type}_{config.num_players}p_{timestamp}.json"
    output_path = config.output_dir / filename

    # Sort by Elo
    sorted_stats = sorted(stats.values(), key=lambda s: s.elo, reverse=True)

    results = {
        "config": {
            "board_type": config.board_type,
            "num_players": config.num_players,
            "games_per_matchup": config.games_per_matchup,
            "device": config.device,
            "think_time_ms": config.think_time_ms,
            "base_seed": config.base_seed,
        },
        "timestamp": timestamp,
        "rankings": [
            {
                "rank": i + 1,
                "agent_id": s.agent_id,
                "elo": round(s.elo, 1),
                "wins": s.wins,
                "losses": s.losses,
                "draws": s.draws,
                "games_played": s.games_played,
                "win_rate": round(s.win_rate, 3),
                "avg_moves": round(s.total_moves / max(1, s.games_played), 1),
            }
            for i, s in enumerate(sorted_stats)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return output_path


def print_results(stats: dict[str, AgentStats]) -> None:
    """Print results table to console."""
    sorted_stats = sorted(stats.values(), key=lambda s: s.elo, reverse=True)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Rank':<6}{'Agent':<20}{'Elo':<10}{'W-L-D':<15}{'Win%':<10}")
    print("-" * 80)

    for i, s in enumerate(sorted_stats):
        wld = f"{s.wins}-{s.losses}-{s.draws}"
        print(f"{i + 1:<6}{s.agent_id:<20}{s.elo:<10.1f}{wld:<15}{s.win_rate * 100:.1f}%")

    print("=" * 80)

    # Highlight top performers
    if sorted_stats:
        best = sorted_stats[0]
        print(f"\nBest performer: {best.agent_id} (Elo: {best.elo:.1f}, Win rate: {best.win_rate * 100:.1f}%)")

        # Elo difference from baseline
        baseline_elo = 1500.0
        for s in sorted_stats:
            if s.agent_id in ("mcts_500", "gumbel_mcts"):
                baseline_elo = s.elo
                break

        elo_gain = best.elo - baseline_elo
        if elo_gain > 0:
            print(f"Elo gain over baseline: +{elo_gain:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI algorithms")
    parser.add_argument("--board", default="square8", help="Board type (square8, square19, hex8, hexagonal)")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players (2-4)")
    parser.add_argument("--games", type=int, default=20, help="Games per matchup")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--model", default=None, help="Neural network model ID")
    parser.add_argument("--think-time", type=int, default=2000, help="Think time in ms")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output", default="data/benchmarks", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Include all algorithms")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer algorithms")

    args = parser.parse_args()

    config = BenchmarkConfig(
        board_type=args.board,
        num_players=args.num_players,
        games_per_matchup=args.games,
        device=args.device,
        nn_model_id=args.model,
        think_time_ms=args.think_time,
        base_seed=args.seed,
        output_dir=Path(args.output),
    )

    if args.quick:
        # Minimal set for quick testing
        config.include_ebmo = False
        config.include_ig_gmo = False
        config.include_gmo_mcts = False
        config.include_improved_mcts = False
    elif not args.full:
        # Default: core algorithms
        config.include_ig_gmo = True
        config.include_gmo_mcts = True

    agent_ids = get_agent_list(config)
    logger.info(f"Benchmarking agents: {agent_ids}")
    logger.info(f"Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"Games per matchup: {config.games_per_matchup}")

    # Run tournament
    stats = run_tournament(agent_ids, config)

    # Print and save results
    print_results(stats)
    output_path = save_results(stats, config)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
