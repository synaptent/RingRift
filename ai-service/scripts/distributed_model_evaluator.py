#!/usr/bin/env python3
"""Distributed Model Evaluator - Automated model pruning system.

This system automatically:
1. Monitors model count across the cluster
2. Triggers distributed gauntlet evaluation when count exceeds threshold
3. Keeps top quartile, archives the rest
4. Runs cyclically as part of the unified AI loop

Usage:
    # Run evaluation cycle
    python scripts/distributed_model_evaluator.py --run

    # Dry run (show what would be pruned)
    python scripts/distributed_model_evaluator.py --dry-run

    # Run as daemon (integrated with unified loop)
    python scripts/distributed_model_evaluator.py --daemon --interval 3600
"""

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import BoardType, AIType, AIConfig, GameStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_COUNT_THRESHOLD = 100
TOP_QUARTILE_KEEP = 0.25
GAMES_PER_BASELINE = 10
PARALLEL_WORKERS = 100  # Default for high-CPU hosts
ARCHIVE_DIR = AI_SERVICE_ROOT / "models" / "archived"
RESULTS_DIR = AI_SERVICE_ROOT / "data" / "evaluations"

# Speed settings
DEFAULT_SEARCH_DEPTH = 4  # For accurate evaluation
FAST_SEARCH_DEPTH = 2  # For quick filtering (~3x faster)

# Cluster nodes for distributed evaluation
CLUSTER_NODES = [
    {"host": "lambda-gh200-a", "cpus": 64, "priority": 1},
    {"host": "lambda-gh200-b", "cpus": 64, "priority": 2},
]


@dataclass
class ModelScore:
    model_path: str
    model_type: str  # "nn" or "nnue"
    board_type: str  # "square8", "square19", "hexagonal"
    vs_random: float
    vs_heuristic: float
    composite_score: float
    games_played: int
    evaluated_at: str


def get_hostname() -> str:
    """Get current hostname."""
    return socket.gethostname()


def detect_board_type(model_path: Path) -> Optional[str]:
    """Detect board type from model filename or sidecar metadata.

    Returns: "square8", "square19", "hexagonal", or None if unknown.
    """
    name = model_path.stem.lower()

    # Try sidecar JSON first (future-proof)
    sidecar = model_path.with_suffix(".json")
    if sidecar.exists():
        try:
            with open(sidecar) as f:
                meta = json.load(f)
                if "board_type" in meta:
                    return meta["board_type"]
        except Exception:
            pass

    # Pattern matching on filename
    if any(p in name for p in ["sq8", "square8", "_8x8"]):
        return "square8"
    elif any(p in name for p in ["sq19", "square19", "_19x19"]):
        return "square19"
    elif any(p in name for p in ["hex", "hexagonal"]):
        return "hexagonal"

    # Default heuristics for common model naming patterns
    if "ringrift_v" in name and "sq" not in name and "hex" not in name:
        # Legacy ringrift_v* models are typically square8
        return "square8"

    return None


def discover_all_models(models_dir: Path, board_filter: Optional[str] = None) -> Dict[str, List[Tuple[Path, str]]]:
    """Discover all NN and NNUE models, grouped by type.

    Args:
        models_dir: Directory to scan
        board_filter: Optional filter like "square8" to only return matching models

    Returns:
        Dict with "nn" and "nnue" keys, each containing list of (path, board_type) tuples
    """
    models = {"nn": [], "nnue": []}

    # NN models (.pth files)
    for f in models_dir.glob("*.pth"):
        if not f.stem.startswith("."):
            board_type = detect_board_type(f)
            if board_type is None:
                logger.warning(f"Unknown board type for {f.name}, skipping")
                continue
            if board_filter and board_type != board_filter:
                continue
            models["nn"].append((f, board_type))

    # NNUE models (.pt files)
    nnue_dir = models_dir / "nnue"
    if nnue_dir.exists():
        for f in nnue_dir.glob("*.pt"):
            if not f.stem.startswith("."):
                board_type = detect_board_type(f)
                if board_type is None:
                    logger.warning(f"Unknown board type for {f.name}, skipping")
                    continue
                if board_filter and board_type != board_filter:
                    continue
                models["nnue"].append((f, board_type))

    return models


def play_single_game(
    model_path: str,
    model_type: str,
    opponent_type: str,
    board_type: BoardType = BoardType.SQUARE8,
    model_plays_first: bool = True,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> Optional[int]:
    """Play a single game, return winner or None for error.

    Args:
        search_depth: Minimax search depth (2=fast, 4=accurate)
    """
    try:
        # Enable neural demo AI for NN model evaluation
        os.environ["AI_ENGINE_NEURAL_DEMO_ENABLED"] = "true"
        # Disable torch.compile for environments without C compiler
        os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"

        from app.game_engine import GameEngine
        from app.main import _create_ai_instance
        from app.training.generate_data import create_initial_state

        state = create_initial_state(board_type=board_type, num_players=2)
        engine = GameEngine()

        # Create opponent
        opp_player = 2 if model_plays_first else 1
        if opponent_type == "random":
            opp_type = AIType.RANDOM
            opp_config = AIConfig(ai_type=opp_type, difficulty=1)
        else:  # heuristic
            opp_type = AIType.HEURISTIC
            opp_config = AIConfig(ai_type=opp_type, difficulty=2)  # Lower for speed
        opponent = _create_ai_instance(opp_type, opp_player, opp_config)

        # Create model AI
        model_player = 1 if model_plays_first else 2
        if model_type == "nnue":
            model_ai_type = AIType.MINIMAX
            model_config = AIConfig(
                ai_type=model_ai_type,
                difficulty=search_depth,
                nnue_model_path=model_path,
            )
        else:
            # NN model - use NeuralNetAI to actually evaluate the neural network
            # Extract model name (stem) from path for nn_model_id
            from pathlib import Path
            model_name = Path(model_path).stem
            model_ai_type = AIType.NEURAL_DEMO
            model_config = AIConfig(
                ai_type=model_ai_type,
                difficulty=5,
                nn_model_id=model_name,
            )
        model_ai = _create_ai_instance(model_ai_type, model_player, model_config)

        ais = {model_player: model_ai, opp_player: opponent}

        for _ in range(300):
            if state.game_status == GameStatus.COMPLETED:
                break
            if hasattr(state.game_status, 'value'):
                if state.game_status.value not in ('active', 'in_progress'):
                    break
            elif str(state.game_status).lower() not in ('active', 'in_progress'):
                break

            ai = ais.get(state.current_player)
            if ai is None:
                break
            move = ai.select_move(state)
            if move is None:
                break
            state = engine.apply_move(state, move)

        return state.winner
    except Exception as e:
        logger.debug(f"Game error: {e}")
        return None


def evaluate_model(
    model_path: str,
    model_type: str,
    board_type_str: str = "square8",
    num_games: int = 10,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> ModelScore:
    """Evaluate a single model against baselines.

    Args:
        model_path: Path to the model file
        model_type: "nn" or "nnue"
        board_type_str: Board type string ("square8", "square19", "hexagonal")
        num_games: Number of games per baseline
        search_depth: Minimax search depth (2=fast ~10s/game, 4=accurate ~45s/game)
    """
    # Convert board type string to enum
    board_type = BoardType(board_type_str)

    wins_random = 0
    wins_heuristic = 0
    games_random = 0
    games_heuristic = 0

    for i in range(num_games):
        model_first = (i % 2 == 0)
        model_player = 1 if model_first else 2

        # vs Random
        winner = play_single_game(
            model_path, model_type, "random",
            board_type=board_type,
            model_plays_first=model_first, search_depth=search_depth
        )
        if winner is not None:
            games_random += 1
            if winner == model_player:
                wins_random += 1

        # vs Heuristic
        winner = play_single_game(
            model_path, model_type, "heuristic",
            board_type=board_type,
            model_plays_first=model_first, search_depth=search_depth
        )
        if winner is not None:
            games_heuristic += 1
            if winner == model_player:
                wins_heuristic += 1

    vs_random = wins_random / games_random if games_random > 0 else 0.0
    vs_heuristic = wins_heuristic / games_heuristic if games_heuristic > 0 else 0.0

    # Weighted composite score
    composite = 2 * vs_heuristic + vs_random

    return ModelScore(
        model_path=model_path,
        model_type=model_type,
        board_type=board_type_str,
        vs_random=vs_random,
        vs_heuristic=vs_heuristic,
        composite_score=composite,
        games_played=games_random + games_heuristic,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def evaluate_model_wrapper(args: Tuple) -> ModelScore:
    """Wrapper for parallel evaluation."""
    model_path, model_type, board_type, num_games, search_depth = args
    return evaluate_model(model_path, model_type, board_type, num_games, search_depth)


def run_distributed_evaluation(
    models: List[Tuple[str, str, str]],  # (path, type, board_type)
    parallel_workers: int = PARALLEL_WORKERS,
    games_per_baseline: int = GAMES_PER_BASELINE,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> List[ModelScore]:
    """Run parallel evaluation across all models.

    Args:
        models: List of (path, model_type, board_type) tuples
        search_depth: Minimax search depth (2=fast, 4=accurate).
                     For 500 models with depth=2: ~30 min
                     For 500 models with depth=4: ~2 hours
    """
    logger.info(f"Evaluating {len(models)} models with {parallel_workers} workers (depth={search_depth})")

    args_list = [(path, mtype, btype, games_per_baseline, search_depth) for path, mtype, btype in models]
    results = []

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(evaluate_model_wrapper, args): i for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"[{len(results)}/{len(models)}] {Path(result.model_path).stem}: "
                           f"score={result.composite_score:.2f} "
                           f"(r={result.vs_random:.0%} h={result.vs_heuristic:.0%})")
            except Exception as e:
                logger.error(f"Failed to evaluate model {idx}: {e}")

    return results


def select_models_to_keep(
    scores: List[ModelScore],
    keep_fraction: float = TOP_QUARTILE_KEEP,
) -> Tuple[List[ModelScore], List[ModelScore]]:
    """Select top models to keep, return (keep, prune) lists."""
    ranked = sorted(scores, key=lambda s: s.composite_score, reverse=True)
    keep_count = max(1, int(len(ranked) * keep_fraction))

    to_keep = ranked[:keep_count]
    to_prune = ranked[keep_count:]

    return to_keep, to_prune


def archive_models(models_to_archive: List[ModelScore], dry_run: bool = False):
    """Move models to archive directory."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_subdir = ARCHIVE_DIR / timestamp

    if not dry_run:
        archive_subdir.mkdir(parents=True, exist_ok=True)

    for score in models_to_archive:
        src = Path(score.model_path)
        if src.exists():
            dst = archive_subdir / src.name
            if dry_run:
                logger.info(f"[DRY RUN] Would archive: {src} -> {dst}")
            else:
                shutil.move(str(src), str(dst))
                logger.info(f"Archived: {src.name}")


def save_evaluation_results(scores: List[ModelScore], keep: List[ModelScore], prune: List[ModelScore]):
    """Save evaluation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"evaluation_{timestamp}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_models": len(scores),
        "kept_count": len(keep),
        "pruned_count": len(prune),
        "all_scores": [asdict(s) for s in scores],
        "kept_models": [s.model_path for s in keep],
        "pruned_models": [s.model_path for s in prune],
    }

    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    return results_file


def run_evaluation_cycle(
    dry_run: bool = False,
    force: bool = False,
    threshold: int = MODEL_COUNT_THRESHOLD,
    workers: int = PARALLEL_WORKERS,
    games: int = GAMES_PER_BASELINE,
    depth: int = DEFAULT_SEARCH_DEPTH,
    board_filter: Optional[str] = None,
):
    """Run a complete evaluation and pruning cycle.

    Args:
        depth: Minimax search depth. Use 2 for fast (~1hr for 500 models),
               4 for accurate (~3hr for 500 models).
        board_filter: Optional filter to only evaluate models for specific board type
                      ("square8", "square19", "hexagonal")
    """
    models_dir = AI_SERVICE_ROOT / "models"

    # Discover models with board type detection
    discovered = discover_all_models(models_dir, board_filter=board_filter)
    total_count = len(discovered["nn"]) + len(discovered["nnue"])

    # Log breakdown by board type
    board_counts = {}
    for path, btype in discovered["nn"] + discovered["nnue"]:
        board_counts[btype] = board_counts.get(btype, 0) + 1
    logger.info(f"Found {total_count} models (NN: {len(discovered['nn'])}, NNUE: {len(discovered['nnue'])})")
    logger.info(f"Board types: {board_counts}")

    # Check threshold
    if total_count < threshold and not force:
        logger.info(f"Model count ({total_count}) below threshold ({threshold}), skipping evaluation")
        return None

    # Estimate time
    games_total = total_count * games * 2  # 2 baselines
    time_per_game = 10 if depth <= 2 else 45  # seconds
    batches = (total_count + workers - 1) // workers
    est_minutes = (batches * games * 2 * time_per_game) / 60
    logger.info(f"Estimated time: {est_minutes:.0f} minutes ({games_total} games, {workers} workers, depth={depth})")

    # Prepare model list with board types
    models = []
    for path, board_type in discovered["nn"]:
        models.append((str(path), "nn", board_type))
    for path, board_type in discovered["nnue"]:
        models.append((str(path), "nnue", board_type))

    # Run evaluation
    scores = run_distributed_evaluation(models, workers, games, depth)

    # Select models
    to_keep, to_prune = select_models_to_keep(scores)

    logger.info(f"Keeping {len(to_keep)} models, pruning {len(to_prune)}")

    # Save results
    results_file = save_evaluation_results(scores, to_keep, to_prune)

    # Archive pruned models
    if to_prune:
        archive_models(to_prune, dry_run=dry_run)

    return results_file


def run_daemon(
    interval_seconds: int = 3600,
    threshold: int = MODEL_COUNT_THRESHOLD,
    workers: int = PARALLEL_WORKERS,
    games: int = GAMES_PER_BASELINE,
    depth: int = FAST_SEARCH_DEPTH,  # Use fast depth by default for daemon
    board_filter: Optional[str] = None,
):
    """Run as daemon, checking periodically."""
    logger.info(f"Starting evaluation daemon (interval: {interval_seconds}s, depth={depth})")

    while True:
        try:
            logger.info("Running scheduled evaluation cycle...")
            run_evaluation_cycle(
                threshold=threshold, workers=workers, games=games,
                depth=depth, board_filter=board_filter
            )
        except Exception as e:
            logger.error(f"Evaluation cycle failed: {e}")

        logger.info(f"Sleeping for {interval_seconds}s until next cycle...")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Model Evaluator - Fast parallel model filtering",
        epilog="""
Examples:
  # Fast evaluation (~30 min for 500 models on 100 workers)
  python scripts/distributed_model_evaluator.py --run --fast --workers 100

  # Accurate evaluation (~2 hours for 500 models on 100 workers)
  python scripts/distributed_model_evaluator.py --run --depth 4 --workers 100

  # Evaluate only square8 models
  python scripts/distributed_model_evaluator.py --run --fast --board square8

  # Run as daemon with fast evaluation every hour
  python scripts/distributed_model_evaluator.py --daemon --fast --workers 100
        """
    )
    parser.add_argument("--run", action="store_true", help="Run evaluation cycle")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned")
    parser.add_argument("--force", action="store_true", help="Force evaluation even below threshold")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=3600, help="Daemon interval in seconds")
    parser.add_argument("--threshold", type=int, default=MODEL_COUNT_THRESHOLD, help="Model count threshold")
    parser.add_argument("--workers", "-j", type=int, default=PARALLEL_WORKERS, help="Parallel workers (default: 100)")
    parser.add_argument("--games", type=int, default=GAMES_PER_BASELINE, help="Games per baseline")
    parser.add_argument("--depth", type=int, default=DEFAULT_SEARCH_DEPTH, help="Minimax search depth (2=fast, 4=accurate)")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (depth=2, ~3x faster)")
    parser.add_argument("--board", type=str, choices=["square8", "square19", "hexagonal"],
                       help="Only evaluate models for specific board type")

    args = parser.parse_args()

    threshold = args.threshold
    workers = args.workers
    games = args.games
    depth = FAST_SEARCH_DEPTH if args.fast else args.depth
    board_filter = args.board

    if args.daemon:
        run_daemon(args.interval, threshold, workers, games, depth, board_filter)
    elif args.run or args.dry_run:
        run_evaluation_cycle(
            dry_run=args.dry_run,
            force=args.force,
            threshold=threshold,
            workers=workers,
            games=games,
            depth=depth,
            board_filter=board_filter,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
