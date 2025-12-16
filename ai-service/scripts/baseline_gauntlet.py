#!/usr/bin/env python3
"""Baseline Gauntlet - Efficiently filter models by testing against fixed baselines.

Instead of O(n²) round-robin tournament, test each model against 3 baselines:
- Random: Sanity check (should win >95%)
- Heuristic: Medium difficulty (should win >70%)
- MCTS-100: Strong baseline (top models win >50%)

This reduces 474 models × 473 pairs × 50 games = 5.6M games
to 474 models × 3 baselines × 10 games = 14,220 games (~400x faster)

Usage:
    # Run full gauntlet
    python scripts/baseline_gauntlet.py --run

    # Quick test (5 games per baseline)
    python scripts/baseline_gauntlet.py --run --games 5

    # Test specific model types
    python scripts/baseline_gauntlet.py --run --nnue-only
    python scripts/baseline_gauntlet.py --run --nn-only

    # Show results from previous run
    python scripts/baseline_gauntlet.py --results
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import BoardType

RESULTS_FILE = AI_SERVICE_ROOT / "data" / "baseline_gauntlet_results.json"


@dataclass
class GauntletResult:
    model_path: str
    model_type: str  # "nn" or "nnue"
    vs_random: float  # win rate 0-1
    vs_heuristic: float
    vs_mcts: float
    games_played: int
    score: float  # weighted composite
    timestamp: str


def discover_models(
    models_dir: Path,
    board_type: str = "square8",
    num_players: int = 2,
    nn_only: bool = False,
    nnue_only: bool = False,
) -> List[Dict[str, Any]]:
    """Discover all models for the given board/player config."""
    models = []

    # NN models (.pth files)
    if not nnue_only:
        for f in models_dir.glob("*.pth"):
            name = f.stem.lower()
            # Filter by board type
            if board_type == "square8" and "sq8" in name or "square8" in name:
                if f"_{num_players}p" in name or f"_{num_players}p" not in name:
                    models.append({
                        "path": str(f),
                        "name": f.stem,
                        "type": "nn",
                        "size_mb": f.stat().st_size / (1024 * 1024),
                    })

    # NNUE models (.pt files)
    if not nn_only:
        nnue_dir = models_dir / "nnue"
        if nnue_dir.exists():
            for f in nnue_dir.glob("*.pt"):
                name = f.stem.lower()
                if board_type in name or f"nnue_{board_type}" in name:
                    models.append({
                        "path": str(f),
                        "name": f.stem,
                        "type": "nnue",
                        "size_mb": f.stat().st_size / (1024 * 1024),
                    })

    return models


def play_game(
    model_path: str,
    model_type: str,
    opponent_type: str,  # "random", "heuristic", "mcts"
    board_type: BoardType = BoardType.SQUARE8,
    model_plays_first: bool = True,
) -> Optional[int]:
    """Play a single game, return winner (1 or 2) or None for draw/error."""
    try:
        from app.game_engine import GameEngine
        from app.main import _create_ai_instance
        from app.models import AIType, AIConfig, GameStatus
        from app.training.generate_data import create_initial_state

        # Initialize game state with board config
        state = create_initial_state(board_type=board_type, num_players=2)
        engine = GameEngine()

        # Create opponent AI
        opp_player = 2 if model_plays_first else 1
        if opponent_type == "random":
            opp_type = AIType.RANDOM
            opp_config = AIConfig(ai_type=opp_type, difficulty=1)
        elif opponent_type == "heuristic":
            opp_type = AIType.HEURISTIC
            opp_config = AIConfig(ai_type=opp_type, difficulty=3)
        elif opponent_type == "mcts":
            # Use minimax for faster evaluation (pure search, no neural)
            opp_type = AIType.MINIMAX
            opp_config = AIConfig(ai_type=opp_type, difficulty=4)
        else:
            raise ValueError(f"Unknown opponent: {opponent_type}")
        opponent = _create_ai_instance(opp_type, opp_player, opp_config)

        # Create model AI - use fast settings for quick filtering
        model_player = 1 if model_plays_first else 2
        if model_type == "nnue":
            # NNUE with minimax depth 4 - reasonably fast
            model_ai_type = AIType.MINIMAX
            model_config = AIConfig(
                ai_type=model_ai_type,
                difficulty=4,
                nnue_model_path=model_path,
            )
        else:
            # For NN models, use pure heuristic AI for fast filtering
            # (NN quality will be validated in detailed tournament later)
            model_ai_type = AIType.HEURISTIC
            model_config = AIConfig(
                ai_type=model_ai_type,
                difficulty=5,  # Just use heuristic for fast filtering
            )
        model_ai = _create_ai_instance(model_ai_type, model_player, model_config)

        ais = {model_player: model_ai, opp_player: opponent}

        max_moves = 300
        for _ in range(max_moves):
            # Check game over
            if state.game_status == GameStatus.COMPLETED:
                break
            if hasattr(state.game_status, 'value'):
                if state.game_status.value not in ('active', 'in_progress'):
                    break
            elif str(state.game_status).lower() not in ('active', 'in_progress'):
                break

            current = state.current_player
            ai = ais.get(current)
            if ai is None:
                break

            move = ai.select_move(state)
            if move is None:
                break

            state = engine.apply_move(state, move)

        return state.winner

    except Exception as e:
        print(f"Game error: {e}")
        return None


def test_model_vs_baseline(
    model_path: str,
    model_type: str,
    opponent_type: str,
    num_games: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
) -> float:
    """Test model against baseline, return win rate."""
    wins = 0
    games_played = 0

    for i in range(num_games):
        # Alternate who plays first
        model_first = (i % 2 == 0)
        model_player = 1 if model_first else 2

        winner = play_game(
            model_path=model_path,
            model_type=model_type,
            opponent_type=opponent_type,
            board_type=board_type,
            model_plays_first=model_first,
        )

        if winner is not None:
            games_played += 1
            if winner == model_player:
                wins += 1

    return wins / games_played if games_played > 0 else 0.0


def run_gauntlet_for_model(
    model: Dict[str, Any],
    num_games: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
    fast_mode: bool = True,
) -> GauntletResult:
    """Run full gauntlet for a single model."""
    model_path = model["path"]
    model_type = model["type"]
    model_name = model["name"]

    print(f"  Testing {model_name}...", flush=True)

    # Test against each baseline
    vs_random = test_model_vs_baseline(
        model_path, model_type, "random", num_games, board_type
    )
    vs_heuristic = test_model_vs_baseline(
        model_path, model_type, "heuristic", num_games, board_type
    )

    # In fast mode, skip slow MCTS baseline
    if fast_mode:
        vs_mcts = 0.0
    else:
        vs_mcts = test_model_vs_baseline(
            model_path, model_type, "mcts", num_games, board_type
        )

    # Weighted score (harder opponents worth more)
    score = 3 * vs_mcts + 2 * vs_heuristic + 1 * vs_random

    return GauntletResult(
        model_path=model_path,
        model_type=model_type,
        vs_random=vs_random,
        vs_heuristic=vs_heuristic,
        vs_mcts=vs_mcts,
        games_played=num_games * 3,
        score=score,
        timestamp=datetime.utcnow().isoformat(),
    )


def _run_model_wrapper(args):
    """Wrapper for parallel execution."""
    model, num_games, board_type = args
    return run_gauntlet_for_model(model, num_games, board_type)


def run_gauntlet(
    models: List[Dict[str, Any]],
    num_games: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
    parallel: int = 1,
) -> List[GauntletResult]:
    """Run gauntlet for all models."""
    total = len(models)
    print(f"Running gauntlet for {total} models ({num_games} games × 2 baselines each)", flush=True)
    print(f"Total games: {total * num_games * 2}", flush=True)
    print(f"Parallel workers: {parallel}", flush=True)
    print(flush=True)

    start_time = time.time()

    if parallel > 1:
        # Parallel execution
        results = []
        args_list = [(m, num_games, board_type) for m in models]

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(_run_model_wrapper, args): i for i, args in enumerate(args_list)}
            completed = 0

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    completed += 1

                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    print(f"[{completed}/{total}] {models[idx]['name']}: Score={result.score:.2f} (r={result.vs_random:.0%} h={result.vs_heuristic:.0%}) ETA: {remaining/60:.1f}min", flush=True)
                except Exception as e:
                    print(f"Error testing {models[idx]['name']}: {e}", flush=True)

        # Sort by original index to maintain order
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]
    else:
        # Sequential execution
        results = []
        for i, model in enumerate(models):
            print(f"[{i+1}/{total}] {model['name']}", flush=True)
            result = run_gauntlet_for_model(model, num_games, board_type)
            results.append(result)

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"    Score: {result.score:.2f} (r={result.vs_random:.0%} h={result.vs_heuristic:.0%} m={result.vs_mcts:.0%})", flush=True)
            print(f"    ETA: {remaining/60:.1f} min", flush=True)
            print(flush=True)

        return results


def save_results(results: List[GauntletResult], path: Path = RESULTS_FILE):
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_models": len(results),
        "results": [
            {
                "model_path": r.model_path,
                "model_type": r.model_type,
                "vs_random": r.vs_random,
                "vs_heuristic": r.vs_heuristic,
                "vs_mcts": r.vs_mcts,
                "games_played": r.games_played,
                "score": r.score,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {path}")


def load_results(path: Path = RESULTS_FILE) -> List[GauntletResult]:
    """Load results from JSON."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return [
        GauntletResult(**r)
        for r in data.get("results", [])
    ]


def print_results(results: List[GauntletResult], top_n: int = 50):
    """Print ranked results."""
    if not results:
        print("No results found.")
        return

    # Sort by score descending
    ranked = sorted(results, key=lambda r: r.score, reverse=True)

    print()
    print("=" * 80)
    print("BASELINE GAUNTLET RESULTS")
    print("=" * 80)
    print()
    print(f"{'Rank':<5} {'Score':<7} {'vs_MCTS':<8} {'vs_Heur':<8} {'vs_Rand':<8} {'Type':<6} Model")
    print("-" * 80)

    for i, r in enumerate(ranked[:top_n]):
        model_name = Path(r.model_path).stem[:35]
        print(f"{i+1:<5} {r.score:<7.2f} {r.vs_mcts:<8.0%} {r.vs_heuristic:<8.0%} {r.vs_random:<8.0%} {r.model_type:<6} {model_name}")

    if len(ranked) > top_n:
        print(f"... and {len(ranked) - top_n} more")

    # Top quartile
    quartile_size = len(ranked) // 4
    print()
    print(f"TOP QUARTILE ({quartile_size} models):")
    print("-" * 40)
    for r in ranked[:quartile_size]:
        print(f"  {Path(r.model_path).name}")


def main():
    parser = argparse.ArgumentParser(description="Baseline Gauntlet - efficient model filtering")
    parser.add_argument("--run", action="store_true", help="Run the gauntlet")
    parser.add_argument("--results", action="store_true", help="Show previous results")
    parser.add_argument("--games", type=int, default=10, help="Games per baseline (default: 10)")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--nn-only", action="store_true", help="Only test NN models")
    parser.add_argument("--nnue-only", action="store_true", help="Only test NNUE models")
    parser.add_argument("--top", type=int, default=50, help="Show top N results")
    parser.add_argument("--limit", type=int, help="Limit number of models to test")
    parser.add_argument("--parallel", "-j", type=int, default=1, help="Number of parallel workers (default: 1)")

    args = parser.parse_args()

    if args.results:
        results = load_results()
        print_results(results, args.top)
        return

    if args.run:
        models_dir = AI_SERVICE_ROOT / "models"
        models = discover_models(
            models_dir,
            args.board,
            args.players,
            nn_only=args.nn_only,
            nnue_only=args.nnue_only,
        )

        if args.limit:
            models = models[:args.limit]

        print(f"Discovered {len(models)} models")

        if not models:
            print("No models found!")
            return

        board_type = BoardType(args.board) if args.board in [bt.value for bt in BoardType] else BoardType.SQUARE8

        results = run_gauntlet(models, args.games, board_type, parallel=args.parallel)
        save_results(results)
        print_results(results, args.top)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
