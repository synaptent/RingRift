#!/usr/bin/env python3
"""Quick benchmark for experimental EBMO/GMO models against baselines."""

import argparse
import sys
from pathlib import Path

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import AIConfig, AIType, BoardType, GameStatus
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI


def run_game(ai1, ai2, board_type, num_players):
    """Run a single game between two AIs. Returns winner (1 or 2) or None for draw.
    ai1 plays as player 1, ai2 plays as player 2 (1-based).
    """
    state = create_initial_state(board_type=board_type, num_players=num_players)
    engine = DefaultRulesEngine()
    ais = {1: ai1, 2: ai2}

    max_moves = 500
    move_count = 0

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current = state.current_player
        ai = ais[current]

        move = ai.select_move(state)
        if move is None:
            break

        state = engine.apply_move(state, move)
        move_count += 1

    if state.game_status == GameStatus.COMPLETED and state.winner is not None:
        return state.winner
    return None


def benchmark_gmo(model_path: Path, num_games: int = 20):
    """Benchmark GMO model against baselines."""
    from app.ai.gmo_ai import GMOAI, GMOConfig

    print(f"\n=== GMO Benchmark: {model_path.name} ===")

    # Create GMO AI (player 1, 1-based)
    gmo_config = GMOConfig(device="cpu")
    ai_config = AIConfig(difficulty=5)
    gmo_ai = GMOAI(1, ai_config, gmo_config)
    gmo_ai.load_checkpoint(model_path)

    # Baselines (player 2, 1-based)
    random_ai = RandomAI(2, AIConfig(difficulty=1))
    heuristic_ai = HeuristicAI(2, AIConfig(difficulty=5))

    # Test vs Random
    wins_vs_random = 0
    for i in range(num_games):
        winner = run_game(gmo_ai, random_ai, BoardType.SQUARE8, 2)
        if winner == 1:  # GMO is player 1
            wins_vs_random += 1
        print(f"\rGMO vs Random: {i+1}/{num_games}", end="")
    print(f"\nGMO vs Random: {wins_vs_random}/{num_games} = {wins_vs_random/num_games*100:.1f}%")

    # Test vs Heuristic
    wins_vs_heur = 0
    for i in range(num_games):
        winner = run_game(gmo_ai, heuristic_ai, BoardType.SQUARE8, 2)
        if winner == 1:  # GMO is player 1
            wins_vs_heur += 1
        print(f"\rGMO vs Heuristic: {i+1}/{num_games}", end="")
    print(f"\nGMO vs Heuristic: {wins_vs_heur}/{num_games} = {wins_vs_heur/num_games*100:.1f}%")

    return wins_vs_random / num_games, wins_vs_heur / num_games


def benchmark_ebmo(model_path: Path, num_games: int = 20):
    """Benchmark EBMO model against baselines."""
    from app.ai.ebmo_ai import EBMO_AI
    from app.ai.ebmo_network import EBMOConfig

    print(f"\n=== EBMO Benchmark: {model_path.name} ===")

    # Create EBMO AI
    ebmo_config = EBMOConfig(
        state_embed_dim=256,
        action_embed_dim=128,
        energy_hidden_dim=256,
        num_energy_layers=3,
        board_size=8,
    )
    ai_config = AIConfig(difficulty=5)
    ebmo_ai = EBMO_AI(1, ai_config, str(model_path), ebmo_config)  # player 1

    # Baselines (player 2)
    random_ai = RandomAI(2, AIConfig(difficulty=1))
    heuristic_ai = HeuristicAI(2, AIConfig(difficulty=5))

    # Test vs Random
    wins_vs_random = 0
    for i in range(num_games):
        winner = run_game(ebmo_ai, random_ai, BoardType.SQUARE8, 2)
        if winner == 1:  # EBMO is player 1
            wins_vs_random += 1
        print(f"\rEBMO vs Random: {i+1}/{num_games}", end="")
    print(f"\nEBMO vs Random: {wins_vs_random}/{num_games} = {wins_vs_random/num_games*100:.1f}%")

    # Test vs Heuristic
    wins_vs_heur = 0
    for i in range(num_games):
        winner = run_game(ebmo_ai, heuristic_ai, BoardType.SQUARE8, 2)
        if winner == 1:  # EBMO is player 1
            wins_vs_heur += 1
        print(f"\rEBMO vs Heuristic: {i+1}/{num_games}", end="")
    print(f"\nEBMO vs Heuristic: {wins_vs_heur}/{num_games} = {wins_vs_heur/num_games*100:.1f}%")

    return wins_vs_random / num_games, wins_vs_heur / num_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmo", type=Path, help="GMO model path")
    parser.add_argument("--ebmo", type=Path, help="EBMO model path")
    parser.add_argument("--games", type=int, default=20, help="Games per matchup")
    args = parser.parse_args()

    results = {}

    if args.gmo:
        r, h = benchmark_gmo(args.gmo, args.games)
        results["gmo"] = {"vs_random": r, "vs_heuristic": h}

    if args.ebmo:
        r, h = benchmark_ebmo(args.ebmo, args.games)
        results["ebmo"] = {"vs_random": r, "vs_heuristic": h}

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for model, scores in results.items():
        print(f"{model.upper()}: Random={scores['vs_random']*100:.0f}%, Heuristic={scores['vs_heuristic']*100:.0f}%")


if __name__ == "__main__":
    main()
