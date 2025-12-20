#!/usr/bin/env python3
"""GMO Hyperparameter Sweep - Find optimal configuration.

Systematically tests different GMO configurations to find the best
hyperparameters for win rate against baselines.

Usage:
    # Full sweep (takes ~2-4 hours on CPU)
    python scripts/gmo_hyperparam_sweep.py --full

    # Quick sweep (subset of parameters)
    python scripts/gmo_hyperparam_sweep.py --quick

    # Single parameter sweep
    python scripts/gmo_hyperparam_sweep.py --param beta --values 0.1,0.3,0.5,1.0

    # Resume from checkpoint
    python scripts/gmo_hyperparam_sweep.py --resume sweep_results.json
"""

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.ai.gmo_ai import GMOAI, GMOConfig
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameStatus
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default checkpoint path
GMO_CHECKPOINT = PROJECT_ROOT / "models" / "gmo" / "gmo_best.pt"
RESULTS_DIR = PROJECT_ROOT / "data" / "gmo_sweep"


@dataclass
class SweepConfig:
    """Configuration for a single sweep trial."""

    # GMO parameters to vary
    optim_steps: int = 10
    lr: float = 0.1
    beta: float = 0.3  # Uncertainty coefficient
    gamma: float = 0.1  # Novelty coefficient
    top_k: int = 5
    mc_samples: int = 10

    # Fixed parameters
    state_dim: int = 128
    move_dim: int = 128
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    novelty_memory_size: int = 1000


@dataclass
class SweepResult:
    """Results from a single sweep trial."""

    config: Dict
    vs_random_wins: int
    vs_random_losses: int
    vs_random_draws: int
    vs_random_winrate: float
    vs_heuristic_wins: int
    vs_heuristic_losses: int
    vs_heuristic_draws: int
    vs_heuristic_winrate: float
    avg_game_length: float
    avg_time_per_move: float
    total_time_sec: float


def create_gmo_from_config(
    sweep_config: SweepConfig,
    player_number: int,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> GMOAI:
    """Create a GMO AI instance from sweep config."""
    from app.models import AIConfig

    # AIConfig for BaseAI
    ai_config = AIConfig(difficulty=5)

    # GMOConfig for GMO-specific parameters
    gmo_config = GMOConfig(
        state_dim=sweep_config.state_dim,
        move_dim=sweep_config.move_dim,
        hidden_dim=sweep_config.hidden_dim,
        top_k=sweep_config.top_k,
        optim_steps=sweep_config.optim_steps,
        lr=sweep_config.lr,
        beta=sweep_config.beta,
        gamma=sweep_config.gamma,
        dropout_rate=sweep_config.dropout_rate,
        mc_samples=sweep_config.mc_samples,
        novelty_memory_size=sweep_config.novelty_memory_size,
        device=device,
    )

    ai = GMOAI(player_number=player_number, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path and checkpoint_path.exists():
        ai.load_checkpoint(checkpoint_path)

    return ai


def play_game(
    ai1,
    ai2,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 500,
) -> Tuple[Optional[int], int, float]:
    """Play a single game and return (winner, num_moves, total_time)."""
    state = create_initial_state(board_type=board_type, num_players=num_players)

    ais = {1: ai1, 2: ai2}
    num_moves = 0
    total_time = 0.0

    for move_num in range(max_moves):
        if state.game_status != GameStatus.ACTIVE:
            break

        current_player = state.current_player
        current_ai = ais[current_player]

        legal_moves = GameEngine.get_valid_moves(state, current_player)
        if not legal_moves:
            # Check for phase requirements (no-action moves)
            phase_req = GameEngine.get_phase_requirement(state, current_player)
            if phase_req:
                bookkeeping_move = GameEngine.synthesize_bookkeeping_move(phase_req, state)
                state = GameEngine.apply_move(state, bookkeeping_move)
                num_moves += 1
                continue
            else:
                break

        start = time.time()
        move = current_ai.select_move(state)
        if move is None:
            break
        total_time += time.time() - start

        state = GameEngine.apply_move(state, move)
        num_moves += 1

    winner = None
    if state.game_status == GameStatus.COMPLETED and state.winner:
        winner = state.winner

    return winner, num_moves, total_time


def evaluate_config(
    sweep_config: SweepConfig,
    games_per_baseline: int = 20,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> SweepResult:
    """Evaluate a single configuration against baselines."""
    start_time = time.time()

    results = {
        "vs_random": {"wins": 0, "losses": 0, "draws": 0},
        "vs_heuristic": {"wins": 0, "losses": 0, "draws": 0},
    }
    total_moves = 0
    total_move_time = 0.0

    baselines = {
        "vs_random": lambda p: RandomAI(player_number=p, config=AIConfig(difficulty=1)),
        "vs_heuristic": lambda p: HeuristicAI(player_number=p, config=AIConfig(difficulty=3)),
    }

    for baseline_name, baseline_factory in baselines.items():
        games_per_side = games_per_baseline // 2

        # GMO as player 1
        for _ in range(games_per_side):
            gmo = create_gmo_from_config(sweep_config, 1, checkpoint_path, device)
            baseline = baseline_factory(2)

            winner, moves, move_time = play_game(gmo, baseline)
            total_moves += moves
            total_move_time += move_time

            if winner == 1:
                results[baseline_name]["wins"] += 1
            elif winner == 2:
                results[baseline_name]["losses"] += 1
            else:
                results[baseline_name]["draws"] += 1

        # GMO as player 2
        for _ in range(games_per_side):
            baseline = baseline_factory(1)
            gmo = create_gmo_from_config(sweep_config, 2, checkpoint_path, device)

            winner, moves, move_time = play_game(baseline, gmo)
            total_moves += moves
            total_move_time += move_time

            if winner == 2:
                results[baseline_name]["wins"] += 1
            elif winner == 1:
                results[baseline_name]["losses"] += 1
            else:
                results[baseline_name]["draws"] += 1

    total_games = games_per_baseline * 2

    return SweepResult(
        config=asdict(sweep_config),
        vs_random_wins=results["vs_random"]["wins"],
        vs_random_losses=results["vs_random"]["losses"],
        vs_random_draws=results["vs_random"]["draws"],
        vs_random_winrate=results["vs_random"]["wins"] / games_per_baseline * 100,
        vs_heuristic_wins=results["vs_heuristic"]["wins"],
        vs_heuristic_losses=results["vs_heuristic"]["losses"],
        vs_heuristic_draws=results["vs_heuristic"]["draws"],
        vs_heuristic_winrate=results["vs_heuristic"]["wins"] / games_per_baseline * 100,
        avg_game_length=total_moves / total_games,
        avg_time_per_move=total_move_time / total_moves if total_moves > 0 else 0,
        total_time_sec=time.time() - start_time,
    )


def generate_sweep_configs(
    param_grid: Dict[str, List],
) -> List[SweepConfig]:
    """Generate all combinations of parameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = []
    for combo in itertools.product(*values):
        config_dict = dict(zip(keys, combo))
        configs.append(SweepConfig(**config_dict))

    return configs


def run_sweep(
    param_grid: Dict[str, List],
    games_per_baseline: int = 20,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
    results_file: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> List[SweepResult]:
    """Run hyperparameter sweep."""
    configs = generate_sweep_configs(param_grid)
    logger.info(f"Generated {len(configs)} configurations to test")

    # Load previous results if resuming
    completed_configs = set()
    results = []
    if resume_from and resume_from.exists():
        with open(resume_from) as f:
            data = json.load(f)
            results = [SweepResult(**r) for r in data.get("results", [])]
            completed_configs = {json.dumps(r.config, sort_keys=True) for r in results}
            logger.info(f"Resuming from {len(results)} completed trials")

    # Run sweep
    for i, config in enumerate(configs):
        config_key = json.dumps(asdict(config), sort_keys=True)
        if config_key in completed_configs:
            logger.info(f"Skipping already completed config {i+1}/{len(configs)}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {i+1}/{len(configs)}")
        logger.info(f"Config: optim_steps={config.optim_steps}, lr={config.lr}, "
                   f"beta={config.beta}, gamma={config.gamma}, top_k={config.top_k}")

        result = evaluate_config(
            config,
            games_per_baseline=games_per_baseline,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        results.append(result)

        logger.info(f"Results: vs_random={result.vs_random_winrate:.1f}%, "
                   f"vs_heuristic={result.vs_heuristic_winrate:.1f}%")
        logger.info(f"Time: {result.total_time_sec:.1f}s")

        # Save checkpoint
        if results_file:
            save_results(results, results_file)

    return results


def save_results(results: List[SweepResult], filepath: Path) -> None:
    """Save sweep results to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_trials": len(results),
        "results": [asdict(r) for r in results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved results to {filepath}")


def analyze_results(results: List[SweepResult]) -> Dict:
    """Analyze sweep results and find best configuration."""
    if not results:
        return {}

    # Sort by combined win rate
    sorted_results = sorted(
        results,
        key=lambda r: (r.vs_heuristic_winrate + r.vs_random_winrate) / 2,
        reverse=True,
    )

    best = sorted_results[0]

    # Analyze parameter importance
    param_importance = {}
    for param in ["optim_steps", "lr", "beta", "gamma", "top_k"]:
        values = set(r.config.get(param) for r in results)
        if len(values) > 1:
            avg_by_value = {}
            for v in values:
                matching = [r for r in results if r.config.get(param) == v]
                avg_winrate = np.mean([
                    (r.vs_heuristic_winrate + r.vs_random_winrate) / 2
                    for r in matching
                ])
                avg_by_value[v] = avg_winrate

            # Importance = range of average winrates
            importance = max(avg_by_value.values()) - min(avg_by_value.values())
            param_importance[param] = {
                "importance": importance,
                "best_value": max(avg_by_value, key=avg_by_value.get),
                "values": avg_by_value,
            }

    return {
        "best_config": best.config,
        "best_vs_random": best.vs_random_winrate,
        "best_vs_heuristic": best.vs_heuristic_winrate,
        "best_combined": (best.vs_heuristic_winrate + best.vs_random_winrate) / 2,
        "param_importance": param_importance,
        "top_5": [
            {
                "config": r.config,
                "vs_random": r.vs_random_winrate,
                "vs_heuristic": r.vs_heuristic_winrate,
            }
            for r in sorted_results[:5]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="GMO Hyperparameter Sweep")

    parser.add_argument(
        "--full", action="store_true",
        help="Run full sweep (all parameter combinations)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick sweep (subset of parameters)"
    )
    parser.add_argument(
        "--param", type=str,
        help="Single parameter to sweep"
    )
    parser.add_argument(
        "--values", type=str,
        help="Comma-separated values for single param sweep"
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Games per baseline per config (default: 20)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=GMO_CHECKPOINT,
        help="Path to GMO checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "sweep_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume from previous results file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--analyze", type=Path,
        help="Analyze existing results file"
    )

    args = parser.parse_args()

    # Analyze mode
    if args.analyze:
        with open(args.analyze) as f:
            data = json.load(f)
            results = [SweepResult(**r) for r in data["results"]]

        analysis = analyze_results(results)
        print("\n" + "="*60)
        print("SWEEP ANALYSIS")
        print("="*60)
        print(f"\nBest Configuration:")
        for k, v in analysis["best_config"].items():
            print(f"  {k}: {v}")
        print(f"\nBest Win Rates:")
        print(f"  vs Random: {analysis['best_vs_random']:.1f}%")
        print(f"  vs Heuristic: {analysis['best_vs_heuristic']:.1f}%")
        print(f"  Combined: {analysis['best_combined']:.1f}%")
        print(f"\nParameter Importance (by win rate range):")
        for param, info in sorted(
            analysis["param_importance"].items(),
            key=lambda x: x[1]["importance"],
            reverse=True
        ):
            print(f"  {param}: {info['importance']:.1f}% range, best={info['best_value']}")
        print(f"\nTop 5 Configurations:")
        for i, cfg in enumerate(analysis["top_5"], 1):
            print(f"  {i}. R:{cfg['vs_random']:.0f}% H:{cfg['vs_heuristic']:.0f}% "
                  f"steps={cfg['config']['optim_steps']} lr={cfg['config']['lr']} "
                  f"beta={cfg['config']['beta']} gamma={cfg['config']['gamma']}")
        return 0

    # Define parameter grids
    if args.full:
        param_grid = {
            "optim_steps": [5, 10, 15, 20],
            "lr": [0.05, 0.1, 0.2, 0.5],
            "beta": [0.1, 0.3, 0.5, 1.0],
            "gamma": [0.0, 0.1, 0.2, 0.5],
            "top_k": [3, 5, 7],
        }
    elif args.quick:
        param_grid = {
            "optim_steps": [5, 10, 20],
            "lr": [0.05, 0.1, 0.2],
            "beta": [0.1, 0.3, 0.5],
            "gamma": [0.0, 0.1, 0.2],
            "top_k": [5],
        }
    elif args.param and args.values:
        values_str = args.values.split(",")
        # Try to parse as numbers
        try:
            values = [float(v) for v in values_str]
            if all(v == int(v) for v in values):
                values = [int(v) for v in values]
        except ValueError:
            values = values_str

        param_grid = {args.param: values}
    else:
        # Default: moderate sweep
        param_grid = {
            "optim_steps": [5, 10, 15],
            "lr": [0.05, 0.1, 0.2],
            "beta": [0.1, 0.3, 0.5],
            "gamma": [0.0, 0.1],
            "top_k": [5],
        }

    logger.info(f"Parameter grid: {param_grid}")
    total_configs = 1
    for v in param_grid.values():
        total_configs *= len(v)
    logger.info(f"Total configurations: {total_configs}")
    logger.info(f"Games per config: {args.games * 2} ({args.games} per baseline)")
    logger.info(f"Estimated total games: {total_configs * args.games * 2}")

    # Run sweep
    results = run_sweep(
        param_grid=param_grid,
        games_per_baseline=args.games,
        checkpoint_path=args.checkpoint,
        device=args.device,
        results_file=args.output,
        resume_from=args.resume,
    )

    # Analyze and print results
    analysis = analyze_results(results)

    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    print(f"\nBest Configuration:")
    for k, v in analysis["best_config"].items():
        print(f"  {k}: {v}")
    print(f"\nBest Win Rates:")
    print(f"  vs Random: {analysis['best_vs_random']:.1f}%")
    print(f"  vs Heuristic: {analysis['best_vs_heuristic']:.1f}%")

    # Save final results
    save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
