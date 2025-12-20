#!/usr/bin/env python3
"""Tune EBMO hyperparameters via grid search."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.INFO)

from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import EBMOConfig
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig
from app.training.initial_state import create_initial_state
from app.game_engine import GameEngine
from app.models.core import BoardType


def play_game(ai1, ai2, max_moves=150):
    """Play a single game and return winner."""
    engine = GameEngine()
    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    ais = {1: ai1, 2: ai2}

    for _ in range(max_moves):
        if state.winner is not None:
            break
        player = state.current_player
        move = ais[player].select_move(state)
        if move is None:
            break
        state = engine.apply_move(state, move)

    return state.winner


def evaluate_config(model_path: str, config: EBMOConfig, num_games: int = 6):
    """Evaluate a configuration against Random and Heuristic."""
    wins_random = 0
    wins_heuristic = 0

    for _i in range(num_games // 2):
        # vs Random
        ai_config = AIConfig(difficulty=5)
        ebmo = EBMO_AI(1, ai_config, model_path=model_path, ebmo_config=config)
        opponent = RandomAI(2, ai_config)
        if play_game(ebmo, opponent) == 1:
            wins_random += 1

        # Swap sides
        ebmo = EBMO_AI(2, ai_config, model_path=model_path, ebmo_config=config)
        opponent = RandomAI(1, ai_config)
        if play_game(opponent, ebmo) == 2:
            wins_random += 1

    for _i in range(num_games // 2):
        # vs Heuristic
        ai_config = AIConfig(difficulty=5)
        ebmo = EBMO_AI(1, ai_config, model_path=model_path, ebmo_config=config)
        opponent = HeuristicAI(2, ai_config)
        if play_game(ebmo, opponent) == 1:
            wins_heuristic += 1

        # Swap sides
        ebmo = EBMO_AI(2, ai_config, model_path=model_path, ebmo_config=config)
        opponent = HeuristicAI(1, ai_config)
        if play_game(opponent, ebmo) == 2:
            wins_heuristic += 1

    return wins_random / num_games, wins_heuristic / num_games


def main():
    model_path = "models/ebmo/ebmo_square8_best.pt"

    # Hyperparameter grid
    skip_penalties = [2.0, 5.0, 10.0]
    optim_steps_list = [50, 100, 150]
    num_restarts_list = [4, 8]
    use_manifold = [True, False]  # False = direct_eval

    print("EBMO Hyperparameter Tuning")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Games per config: 6 (3 vs Random, 3 vs Heuristic)")
    print()

    results = []

    # First test current baseline
    baseline_config = EBMOConfig()
    print("Testing baseline config...")
    rand_wr, heur_wr = evaluate_config(model_path, baseline_config)
    print(f"Baseline: skip={baseline_config.skip_penalty}, steps={baseline_config.optim_steps}, "
          f"restarts={baseline_config.num_restarts}, manifold={baseline_config.use_manifold_optim}")
    print(f"  -> Random: {rand_wr*100:.0f}%, Heuristic: {heur_wr*100:.0f}%, "
          f"Avg: {(rand_wr+heur_wr)*50:.0f}%")
    results.append((baseline_config, rand_wr, heur_wr))
    print()

    # Grid search (reduced for speed)
    configs_to_test = [
        # Vary skip penalty
        (2.0, 100, 8, True),
        (10.0, 100, 8, True),
        # Vary optim steps
        (5.0, 50, 8, True),
        (5.0, 150, 8, True),
        # Vary restarts
        (5.0, 100, 4, True),
        (5.0, 100, 12, True),
        # Direct eval (no optimization)
        (5.0, 0, 0, False),
        # Combined best guesses
        (3.0, 80, 6, True),
        (8.0, 120, 10, True),
    ]

    for skip, steps, restarts, manifold in configs_to_test:
        config = EBMOConfig(
            skip_penalty=skip,
            optim_steps=steps,
            num_restarts=restarts,
            use_manifold_optim=manifold,
            use_direct_eval=not manifold,
        )

        print(f"Testing: skip={skip}, steps={steps}, restarts={restarts}, manifold={manifold}")
        rand_wr, heur_wr = evaluate_config(model_path, config)
        avg = (rand_wr + heur_wr) / 2
        print(f"  -> Random: {rand_wr*100:.0f}%, Heuristic: {heur_wr*100:.0f}%, Avg: {avg*100:.0f}%")
        results.append((config, rand_wr, heur_wr))

    # Find best config
    print()
    print("=" * 70)
    print("Results Summary (sorted by average win rate)")
    print("=" * 70)

    results.sort(key=lambda x: -(x[1] + x[2]))

    for i, (cfg, rand_wr, heur_wr) in enumerate(results[:5]):
        avg = (rand_wr + heur_wr) / 2
        print(f"{i+1}. skip={cfg.skip_penalty}, steps={cfg.optim_steps}, "
              f"restarts={cfg.num_restarts}, manifold={cfg.use_manifold_optim}")
        print(f"   Random: {rand_wr*100:.0f}%, Heuristic: {heur_wr*100:.0f}%, Avg: {avg*100:.0f}%")

    # Output best config
    best = results[0]
    print()
    print("Best configuration:")
    print(f"  skip_penalty = {best[0].skip_penalty}")
    print(f"  optim_steps = {best[0].optim_steps}")
    print(f"  num_restarts = {best[0].num_restarts}")
    print(f"  use_manifold_optim = {best[0].use_manifold_optim}")


if __name__ == "__main__":
    main()
