#!/usr/bin/env python3
"""Estimate Elo ratings for AI players through round-robin tournament.

Usage:
    python scripts/estimate_elo.py --num-games 10 --output results/elo_ratings.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.policy_only_ai import PolicyOnlyAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Tournament player."""
    name: str
    ai_type: str
    config: dict[str, Any]
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0


@dataclass
class MatchResult:
    """Result of a single match."""
    player1: str
    player2: str
    winner: int | None  # 1, 2, or None for draw
    moves: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EloRating:
    """Elo rating system."""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, player: str) -> float:
        return self.ratings.get(player, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, player_a: str, player_b: str, score_a: float) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            player_a: First player name
            player_b: Second player name
            score_a: Score for player A (1.0 win, 0.5 draw, 0.0 loss)

        Returns:
            Tuple of new ratings (rating_a, rating_b)
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        score_b = 1.0 - score_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b

        return new_rating_a, new_rating_b


def create_ai(player_number: int, player: Player, board_type: BoardType):
    """Create an AI instance for a player."""
    config = AIConfig(**player.config)

    if player.ai_type == "heuristic":
        return HeuristicAI(player_number=player_number, config=config)
    elif player.ai_type == "policy_only":
        return PolicyOnlyAI(player_number=player_number, config=config, board_type=board_type)
    elif player.ai_type == "gumbel_mcts":
        return GumbelMCTSAI(player_number=player_number, config=config, board_type=board_type)
    else:
        raise ValueError(f"Unknown AI type: {player.ai_type}")


def play_game(
    p1: Player,
    p2: Player,
    board_type: BoardType,
    max_moves: int = 300,
    seed: int | None = None,
) -> MatchResult:
    """Play a single game between two players."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state = create_initial_state(board_type, 2)
    engine = GameEngine()

    p1_ai = create_ai(1, p1, board_type)
    p2_ai = create_ai(2, p2, board_type)

    move_count = 0

    for _ in range(max_moves):
        status = state.game_status.value
        if status not in ['active', 'in_progress']:
            break

        current = state.current_player
        ai = p1_ai if current == 1 else p2_ai

        move = ai.select_move(state)
        if move is None:
            break

        state = engine.apply_move(state, move)
        move_count += 1

    return MatchResult(
        player1=p1.name,
        player2=p2.name,
        winner=state.winner,
        moves=move_count,
    )


def run_tournament(
    players: list[Player],
    num_games_per_pair: int,
    board_type: BoardType,
    elo_system: EloRating,
) -> list[MatchResult]:
    """Run round-robin tournament."""
    results = []
    total_matches = len(players) * (len(players) - 1) * num_games_per_pair
    completed = 0

    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if i >= j:
                continue  # Skip self and duplicates

            for game_num in range(num_games_per_pair):
                # Alternate colors
                if game_num % 2 == 0:
                    white, black = p1, p2
                else:
                    white, black = p2, p1

                seed = 42 + i * 1000 + j * 100 + game_num
                result = play_game(white, black, board_type, seed=seed)
                results.append(result)

                # Update stats
                if result.winner == 1:
                    white.wins += 1
                    black.losses += 1
                    score = 1.0
                elif result.winner == 2:
                    white.losses += 1
                    black.wins += 1
                    score = 0.0
                else:
                    white.draws += 1
                    black.draws += 1
                    score = 0.5

                # Update Elo
                elo_system.update_ratings(white.name, black.name, score)

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total_matches} matches")

    return results


def main():
    parser = argparse.ArgumentParser(description="Estimate Elo ratings for AI players")
    parser.add_argument("--num-games", type=int, default=10, help="Games per pair")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--output", type=str, default="results/elo_ratings.json", help="Output file")
    parser.add_argument("--all-models", action="store_true", help="Test all model variants")
    args = parser.parse_args()

    board_type = BoardType.SQUARE8 if "square8" in args.board.lower() else BoardType.HEXAGONAL

    # Define players
    if args.all_models:
        players = [
            Player(
                name="heuristic",
                ai_type="heuristic",
                config={"difficulty": 5},
            ),
            Player(
                name="policy_v1",
                ai_type="policy_only",
                config={
                    "difficulty": 5,
                    "nn_model_id": "distilled_sq8_2p",
                    "temperature": 0.5,
                },
            ),
            Player(
                name="policy_v3",
                ai_type="policy_only",
                config={
                    "difficulty": 5,
                    "nn_model_id": "distilled_sq8_2p_v3",
                    "temperature": 0.5,
                },
            ),
            Player(
                name="gumbel_v1",
                ai_type="gumbel_mcts",
                config={
                    "difficulty": 5,
                    "nn_model_id": "distilled_sq8_2p",
                    "think_time": 500,
                    "num_simulations": 50,
                },
            ),
            Player(
                name="gumbel_v3",
                ai_type="gumbel_mcts",
                config={
                    "difficulty": 5,
                    "nn_model_id": "distilled_sq8_2p_v3",
                    "think_time": 500,
                    "num_simulations": 50,
                },
            ),
        ]
    else:
        players = [
            Player(
                name="heuristic",
                ai_type="heuristic",
                config={"difficulty": 5},
            ),
            Player(
                name="policy_distilled",
                ai_type="policy_only",
                config={
                    "difficulty": 5,
                    "nn_model_id": "distilled_sq8_2p",
                    "temperature": 0.5,
                },
            ),
            Player(
                name="gumbel_distilled",
                ai_type="gumbel_mcts",
                config={
                    "difficulty": 5,
                    "nn_model_id": "distilled_sq8_2p",
                    "think_time": 500,
                    "num_simulations": 50,
                },
            ),
        ]

    elo_system = EloRating(k_factor=32.0)

    logger.info(f"Starting tournament with {len(players)} players")
    logger.info(f"Games per pair: {args.num_games}")

    results = run_tournament(players, args.num_games, board_type, elo_system)

    # Compile results
    output = {
        "timestamp": datetime.now().isoformat(),
        "board_type": board_type.value,
        "games_per_pair": args.num_games,
        "players": [],
        "matches": [
            {
                "player1": r.player1,
                "player2": r.player2,
                "winner": r.winner,
                "moves": r.moves,
            }
            for r in results
        ],
    }

    for p in sorted(players, key=lambda x: elo_system.get_rating(x.name), reverse=True):
        output["players"].append({
            "name": p.name,
            "elo": round(elo_system.get_rating(p.name)),
            "wins": p.wins,
            "losses": p.losses,
            "draws": p.draws,
            "win_rate": round(p.win_rate * 100, 1),
        })

    # Print summary
    logger.info("=" * 60)
    logger.info("TOURNAMENT RESULTS")
    logger.info("=" * 60)
    for p in output["players"]:
        logger.info(f"{p['name']}: Elo={p['elo']}, W/L/D={p['wins']}/{p['losses']}/{p['draws']} ({p['win_rate']}%)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
