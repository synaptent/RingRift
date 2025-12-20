#!/usr/bin/env python3
"""GMO Ablation Study - Measure contribution of each component.

This script systematically disables each GMO component to measure
its contribution to overall performance.

Ablations tested:
1. No gradient optimization (just use initial ranking)
2. No uncertainty bonus (beta=0)
3. No novelty bonus (gamma=0)
4. No MC Dropout (single forward pass)
5. Fewer optimization steps (1 vs 10)
6. Smaller top-k (1 vs 5)

Usage:
    python scripts/gmo_ablation_study.py --games 30
    python scripts/gmo_ablation_study.py --ablation no_uncertainty --games 50
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.ai.gmo_ai import GMOAI, GMOConfig
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GMO_CHECKPOINT = PROJECT_ROOT / "models" / "gmo" / "gmo_best.pt"
RESULTS_DIR = PROJECT_ROOT / "data" / "gmo_ablation"


@dataclass
class AblationConfig:
    """Configuration for an ablation variant."""
    name: str
    description: str
    optim_steps: int = 10
    lr: float = 0.1
    beta: float = 0.3
    gamma: float = 0.1
    top_k: int = 5
    mc_samples: int = 10
    use_gradient_optim: bool = True


# Define ablation variants
ABLATIONS = {
    "full": AblationConfig(
        name="full",
        description="Full GMO (baseline)",
    ),
    "no_gradient": AblationConfig(
        name="no_gradient",
        description="No gradient optimization - just initial ranking",
        use_gradient_optim=False,
        optim_steps=0,
    ),
    "no_uncertainty": AblationConfig(
        name="no_uncertainty",
        description="No uncertainty bonus (beta=0)",
        beta=0.0,
    ),
    "no_novelty": AblationConfig(
        name="no_novelty",
        description="No novelty bonus (gamma=0)",
        gamma=0.0,
    ),
    "single_forward": AblationConfig(
        name="single_forward",
        description="Single forward pass (no MC Dropout)",
        mc_samples=1,
    ),
    "fewer_steps": AblationConfig(
        name="fewer_steps",
        description="Fewer optimization steps (1 vs 10)",
        optim_steps=1,
    ),
    "single_candidate": AblationConfig(
        name="single_candidate",
        description="Single candidate (top_k=1)",
        top_k=1,
    ),
    "no_exploration": AblationConfig(
        name="no_exploration",
        description="No exploration (beta=0, gamma=0)",
        beta=0.0,
        gamma=0.0,
    ),
}


class AblatedGMOAI(GMOAI):
    """GMO AI with ablation support."""

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        ablation_config: AblationConfig,
        device: str = "cpu",
    ):
        gmo_config = GMOConfig(
            optim_steps=ablation_config.optim_steps,
            lr=ablation_config.lr,
            beta=ablation_config.beta,
            gamma=ablation_config.gamma,
            top_k=ablation_config.top_k,
            mc_samples=ablation_config.mc_samples,
            device=device,
        )
        super().__init__(player_number, config, gmo_config)
        self.ablation_config = ablation_config

    def select_move(self, game_state):
        """Select move with ablation applied."""
        if not self.ablation_config.use_gradient_optim:
            # Skip gradient optimization - just use initial ranking
            return self._select_move_no_gradient(game_state)
        return super().select_move(game_state)

    def _select_move_no_gradient(self, game_state):
        """Select move using only initial value estimates."""
        legal_moves = GameEngine.get_valid_moves(game_state, game_state.current_player)

        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        import torch

        # Encode state using the proper method
        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

        # Score all moves
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            with torch.no_grad():
                move_embed = self.move_encoder.encode_move(move)
                value, log_var = self.value_net(state_embed, move_embed)

            score = value.item()
            if self.gmo_config.beta > 0:
                var = torch.exp(log_var).item()
                score += self.gmo_config.beta * np.sqrt(var)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


def create_ablated_gmo(
    player_number: int,
    ablation_config: AblationConfig,
    device: str = "cpu",
) -> AblatedGMOAI:
    """Create GMO AI with specific ablation."""
    ai_config = AIConfig(difficulty=5)
    ai = AblatedGMOAI(
        player_number=player_number,
        config=ai_config,
        ablation_config=ablation_config,
        device=device,
    )
    if GMO_CHECKPOINT.exists():
        ai.load_checkpoint(GMO_CHECKPOINT)
    return ai


def play_game(ai1, ai2, max_moves: int = 500) -> tuple[int | None, int]:
    """Play a game and return (winner, num_moves)."""
    from app.training.initial_state import create_initial_state

    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    ais = {1: ai1, 2: ai2}

    for _move_num in range(max_moves):
        if state.game_status != GameStatus.ACTIVE:
            break

        current_player = state.current_player
        current_ai = ais[current_player]
        legal_moves = GameEngine.get_valid_moves(state, current_player)

        if not legal_moves:
            # Check for phase requirements (no-action moves)
            phase_req = GameEngine.get_phase_requirement(state, current_player)
            if phase_req:
                # Create and apply the bookkeeping move
                bookkeeping_move = GameEngine.synthesize_bookkeeping_move(phase_req, state)
                state = GameEngine.apply_move(state, bookkeeping_move)
                continue
            else:
                # No legal moves and no phase requirement - game is stuck
                break

        move = current_ai.select_move(state)
        if move is None:
            # AI couldn't select a move
            break
        state = GameEngine.apply_move(state, move)

    winner = state.winner if state.game_status == GameStatus.COMPLETED else None
    return winner, move_num + 1


def run_ablation(
    ablation_name: str,
    num_games: int = 30,
    device: str = "cpu",
) -> dict:
    """Run a single ablation study."""
    ablation_config = ABLATIONS[ablation_name]
    logger.info(f"Running ablation: {ablation_name}")
    logger.info(f"  Description: {ablation_config.description}")

    results = {
        "vs_random": {"wins": 0, "losses": 0, "draws": 0},
        "vs_heuristic": {"wins": 0, "losses": 0, "draws": 0},
    }
    total_moves = 0
    start_time = time.time()

    baselines = {
        "vs_random": lambda p: RandomAI(player_number=p, config=AIConfig(difficulty=1)),
        "vs_heuristic": lambda p: HeuristicAI(player_number=p, config=AIConfig(difficulty=3)),
    }

    games_per_baseline = num_games // 2

    for baseline_name, baseline_factory in baselines.items():
        # GMO as P1
        for _ in range(games_per_baseline // 2):
            gmo = create_ablated_gmo(1, ablation_config, device)
            baseline = baseline_factory(2)

            winner, moves = play_game(gmo, baseline)
            total_moves += moves

            if winner == 1:
                results[baseline_name]["wins"] += 1
            elif winner == 2:
                results[baseline_name]["losses"] += 1
            else:
                results[baseline_name]["draws"] += 1

        # GMO as P2
        for _ in range(games_per_baseline // 2):
            baseline = baseline_factory(1)
            gmo = create_ablated_gmo(2, ablation_config, device)

            winner, moves = play_game(baseline, gmo)
            total_moves += moves

            if winner == 2:
                results[baseline_name]["wins"] += 1
            elif winner == 1:
                results[baseline_name]["losses"] += 1
            else:
                results[baseline_name]["draws"] += 1

    total_games = num_games
    elapsed = time.time() - start_time

    return {
        "ablation": ablation_name,
        "description": ablation_config.description,
        "config": {
            "optim_steps": ablation_config.optim_steps,
            "beta": ablation_config.beta,
            "gamma": ablation_config.gamma,
            "top_k": ablation_config.top_k,
            "mc_samples": ablation_config.mc_samples,
            "use_gradient_optim": ablation_config.use_gradient_optim,
        },
        "vs_random_winrate": results["vs_random"]["wins"] / (games_per_baseline) * 100,
        "vs_heuristic_winrate": results["vs_heuristic"]["wins"] / (games_per_baseline) * 100,
        "vs_random": results["vs_random"],
        "vs_heuristic": results["vs_heuristic"],
        "avg_game_length": total_moves / total_games,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="GMO Ablation Study")
    parser.add_argument(
        "--ablation", type=str, default=None,
        help=f"Specific ablation to run (choices: {list(ABLATIONS.keys())})"
    )
    parser.add_argument(
        "--games", type=int, default=30,
        help="Games per ablation"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "ablation_results.json",
        help="Output file"
    )

    args = parser.parse_args()

    ablations_to_run = [args.ablation] if args.ablation else list(ABLATIONS.keys())
    results = []

    print("\n" + "="*70)
    print("GMO ABLATION STUDY")
    print("="*70)

    for ablation_name in ablations_to_run:
        if ablation_name not in ABLATIONS:
            logger.error(f"Unknown ablation: {ablation_name}")
            continue

        result = run_ablation(ablation_name, args.games, args.device)
        results.append(result)

        print(f"\n{ablation_name}:")
        print(f"  {result['description']}")
        print(f"  vs Random: {result['vs_random_winrate']:.1f}%")
        print(f"  vs Heuristic: {result['vs_heuristic_winrate']:.1f}%")

    # Summary table
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"\n{'Ablation':<20} {'vs Random':>12} {'vs Heuristic':>14} {'Delta':>10}")
    print("-"*60)

    # Find baseline (full GMO) performance
    baseline = next((r for r in results if r["ablation"] == "full"), None)
    baseline_combined = 0
    if baseline:
        baseline_combined = (baseline["vs_random_winrate"] + baseline["vs_heuristic_winrate"]) / 2

    for r in results:
        combined = (r["vs_random_winrate"] + r["vs_heuristic_winrate"]) / 2
        delta = combined - baseline_combined if baseline else 0
        delta_str = f"{delta:+.1f}%" if baseline else "N/A"

        print(f"{r['ablation']:<20} {r['vs_random_winrate']:>11.1f}% {r['vs_heuristic_winrate']:>13.1f}% {delta_str:>10}")

    # Component importance
    if baseline and len(results) > 1:
        print("\n" + "-"*60)
        print("\nComponent Importance (performance drop when removed):")

        importance = []
        for r in results:
            if r["ablation"] != "full":
                combined = (r["vs_random_winrate"] + r["vs_heuristic_winrate"]) / 2
                drop = baseline_combined - combined
                importance.append((r["ablation"], drop, r["description"]))

        importance.sort(key=lambda x: x[1], reverse=True)
        for name, drop, desc in importance:
            print(f"  {name:<20}: {drop:+.1f}% ({desc})")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "games_per_ablation": args.games,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
