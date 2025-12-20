#!/usr/bin/env python
"""Select the best checkpoint based on Elo evaluation with baseline gating.

This script addresses the loss/Elo disconnect where lower validation loss
doesn't always correlate with better playing strength. It:

1. Finds all checkpoints for a training run (epoch checkpoints + final)
2. Runs a mini-gauntlet (fast games against random/heuristic)
3. **GATES** checkpoints that don't meet minimum win rates vs baselines
4. Selects the checkpoint with highest estimated Elo among passing checkpoints
5. Copies it as the "best" checkpoint

IMPORTANT: Checkpoints must beat random at MIN_RANDOM_WIN_RATE (default 85%)
and heuristic at MIN_HEURISTIC_WIN_RATE (default 60%) to be considered.
This prevents selecting checkpoints that are strong in neural-vs-neural
but weak against basic baselines.

Usage:
    python scripts/select_best_checkpoint_by_elo.py \
        --candidate-id sq8_2p_d8_cand_20251218_040151 \
        --games 20

    # Custom thresholds
    python scripts/select_best_checkpoint_by_elo.py \
        --candidate-id sq8_2p_d10_cand_20251218_171759 \
        --min-random-win-rate 0.90 \
        --min-heuristic-win-rate 0.70 \
        --games 30
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Only import what we need directly - game_gauntlet handles the rest
from app.models import BoardType, AIType, AIConfig
from app.ai.policy_only_ai import PolicyOnlyAI

# Import canonical thresholds from single source of truth
try:
    from app.config.thresholds import (
        MIN_WIN_RATE_VS_RANDOM,
        MIN_WIN_RATE_VS_HEURISTIC,
    )
    DEFAULT_MIN_RANDOM_WIN_RATE = MIN_WIN_RATE_VS_RANDOM
    DEFAULT_MIN_HEURISTIC_WIN_RATE = MIN_WIN_RATE_VS_HEURISTIC
except ImportError:
    # Fallback if running standalone
    DEFAULT_MIN_RANDOM_WIN_RATE = 0.85
    DEFAULT_MIN_HEURISTIC_WIN_RATE = 0.60


def is_versioned_checkpoint(checkpoint_path: Path) -> bool:
    """Check if a checkpoint has versioning metadata.

    Versioned checkpoints are safer to load as they include architecture info.
    """
    try:
        import torch
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return "version" in data or "__version__" in data or "architecture" in data
    except Exception:
        return False


def find_checkpoints(
    candidate_id: str,
    models_dir: str = "models",
    skip_checkpoint_dir: bool = True,
    versioned_only: bool = True,
) -> list[Path]:
    """Find all checkpoints for a candidate model.

    Looks for:
    - {candidate_id}.pth (final/best by loss)
    - {candidate_id}_*.pth (epoch checkpoints)
    - optionally checkpoints/{candidate_id}/*.pth (disabled by default)

    Args:
        candidate_id: Model candidate ID prefix
        models_dir: Base models directory
        skip_checkpoint_dir: If True, skip checkpoints/ subdirectory (default True)
            These legacy checkpoints often have architecture mismatches.
        versioned_only: If True, only include versioned checkpoints (default True)
    """
    models_path = Path(models_dir).resolve()
    checkpoints = []

    # Main checkpoint
    main_ckpt = models_path / f"{candidate_id}.pth"
    if main_ckpt.exists():
        checkpoints.append(main_ckpt.resolve())

    # Epoch checkpoints with timestamps
    for f in models_path.glob(f"{candidate_id}_*.pth"):
        if f.exists() and "_elo_best" not in f.name:  # Skip elo_best to avoid circular
            checkpoints.append(f.resolve())

    # Checkpoint directory (disabled by default due to legacy architecture issues)
    if not skip_checkpoint_dir:
        ckpt_dir = models_path / "checkpoints" / candidate_id
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*.pth"):
                checkpoints.append(f.resolve())

    # Filter to versioned-only if requested
    if versioned_only:
        original_count = len(checkpoints)
        checkpoints = [c for c in checkpoints if is_versioned_checkpoint(c)]
        skipped = original_count - len(checkpoints)
        if skipped > 0:
            print(f"  Skipped {skipped} legacy (unversioned) checkpoints")

    return sorted(checkpoints, key=lambda x: x.stat().st_mtime)


class CheckpointLoadError(Exception):
    """Raised when a checkpoint cannot be loaded due to architecture mismatch."""


def validate_checkpoint_loadable(
    checkpoint_path: Path,
    board_type: BoardType,
) -> bool:
    """Test if a checkpoint can be loaded without architecture errors.

    This performs a quick validation by attempting to load the model.
    Returns True if loadable, False otherwise.
    """
    try:
        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_model_id=str(checkpoint_path),
            policy_temperature=0.5,
        )
        # Try to create the AI - this will fail if architecture mismatches
        ai = PolicyOnlyAI(1, config, board_type=board_type)
        # Try to get a move to ensure model is fully loaded
        return True
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Error(s) in loading state_dict" in str(e):
            return False
        raise
    except Exception:
        return False


def evaluate_checkpoint(
    checkpoint_path: Path,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    games_per_opponent: int = 10,
) -> dict[str, Any]:
    """Evaluate a checkpoint via mini-gauntlet against baselines.

    Uses the unified game_gauntlet module for consistent evaluation.

    Returns dict with win rates and estimated Elo.
    Raises CheckpointLoadError if checkpoint cannot be loaded.
    """
    from app.training.game_gauntlet import (
        run_baseline_gauntlet,
        BaselineOpponent,
    )

    # Pre-validate checkpoint is loadable
    if not validate_checkpoint_loadable(checkpoint_path, board_type):
        raise CheckpointLoadError(
            f"Cannot load checkpoint {checkpoint_path}: architecture mismatch"
        )

    # Use unified game gauntlet for evaluation
    gauntlet_result = run_baseline_gauntlet(
        model_path=checkpoint_path,
        board_type=board_type,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        games_per_opponent=games_per_opponent,
        num_players=num_players,
        check_baseline_gating=False,  # We do gating at select_best level
        verbose=False,
    )

    # Convert to legacy result format for backwards compatibility
    results = {
        "checkpoint": str(checkpoint_path),
        "games": gauntlet_result.total_games,
        "wins": gauntlet_result.total_wins,
        "losses": gauntlet_result.total_losses,
        "draws": gauntlet_result.total_draws,
        "win_rate": gauntlet_result.win_rate,
        "estimated_elo": gauntlet_result.estimated_elo,
        "vs_random": gauntlet_result.opponent_results.get("random", {"wins": 0, "games": 0, "win_rate": 0.0}),
        "vs_heuristic": gauntlet_result.opponent_results.get("heuristic", {"wins": 0, "games": 0, "win_rate": 0.0}),
    }

    return results


def select_best_checkpoint(
    candidate_id: str,
    models_dir: str = "models",
    games_per_opponent: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    min_random_win_rate: float = DEFAULT_MIN_RANDOM_WIN_RATE,
    min_heuristic_win_rate: float = DEFAULT_MIN_HEURISTIC_WIN_RATE,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Evaluate all checkpoints and select the best one by Elo.

    Checkpoints must pass baseline gating requirements before being
    considered for Elo-based selection:
    - Must beat random at min_random_win_rate
    - Must beat heuristic at min_heuristic_win_rate

    Returns (best_checkpoint_path, all_results).
    """
    checkpoints = find_checkpoints(candidate_id, models_dir)

    if not checkpoints:
        print(f"No checkpoints found for {candidate_id}")
        return None, []

    print(f"Found {len(checkpoints)} checkpoints for {candidate_id}")
    print(f"Baseline requirements: random >= {min_random_win_rate:.0%}, heuristic >= {min_heuristic_win_rate:.0%}")

    all_results = []
    qualified_checkpoints = []  # Checkpoints that pass baseline gating
    best_elo = float("-inf")
    best_checkpoint = None

    for i, ckpt in enumerate(checkpoints):
        print(f"\nEvaluating [{i+1}/{len(checkpoints)}] {ckpt.name}...")

        try:
            result = evaluate_checkpoint(
                ckpt,
                board_type=board_type,
                num_players=num_players,
                games_per_opponent=games_per_opponent,
            )
            all_results.append(result)

            vs_random = result['vs_random']['win_rate']
            vs_heuristic = result['vs_heuristic']['win_rate']

            print(f"  Win rate: {result['win_rate']:.1%}")
            print(f"  vs Random: {vs_random:.1%}", end="")
            if vs_random < min_random_win_rate:
                print(f" [BELOW {min_random_win_rate:.0%} THRESHOLD]")
            else:
                print(" [OK]")

            print(f"  vs Heuristic: {vs_heuristic:.1%}", end="")
            if vs_heuristic < min_heuristic_win_rate:
                print(f" [BELOW {min_heuristic_win_rate:.0%} THRESHOLD]")
            else:
                print(" [OK]")

            print(f"  Estimated Elo: {result['estimated_elo']:.0f}")

            # Check baseline gating
            passes_gating = (
                vs_random >= min_random_win_rate and
                vs_heuristic >= min_heuristic_win_rate
            )

            if passes_gating:
                result["qualified"] = True
                qualified_checkpoints.append((ckpt, result))
                print("  STATUS: QUALIFIED")

                if result["estimated_elo"] > best_elo:
                    best_elo = result["estimated_elo"]
                    best_checkpoint = ckpt
            else:
                result["qualified"] = False
                print("  STATUS: DISQUALIFIED (below baseline thresholds)")

        except CheckpointLoadError as e:
            print(f"  SKIPPED: {e}")
            continue

        except Exception as e:
            print(f"  Error evaluating {ckpt.name}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(qualified_checkpoints)}/{len(all_results)} checkpoints qualified")

    if not qualified_checkpoints:
        print("WARNING: No checkpoints passed baseline gating!")
        print("Consider:")
        print("  1. Lowering thresholds (--min-random-win-rate, --min-heuristic-win-rate)")
        print("  2. Training longer or with more diverse data")
        print("  3. Checking for training issues (data quality, hyperparameters)")

        # Fall back to loss-best (original checkpoint) if no qualified checkpoints
        main_ckpt = Path(models_dir) / f"{candidate_id}.pth"
        if main_ckpt.exists():
            print(f"\nFalling back to loss-best checkpoint: {main_ckpt.name}")
            return main_ckpt, all_results

    return best_checkpoint, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Select best checkpoint by Elo evaluation with baseline gating"
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Candidate model ID to evaluate",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Games per opponent for evaluation",
    )
    parser.add_argument(
        "--board-type",
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players",
    )
    parser.add_argument(
        "--copy-best",
        action="store_true",
        help="Copy best checkpoint to {candidate_id}_best.pth",
    )
    parser.add_argument(
        "--min-random-win-rate",
        type=float,
        default=DEFAULT_MIN_RANDOM_WIN_RATE,
        help=f"Minimum win rate vs random to qualify (default: {DEFAULT_MIN_RANDOM_WIN_RATE:.0%})",
    )
    parser.add_argument(
        "--min-heuristic-win-rate",
        type=float,
        default=DEFAULT_MIN_HEURISTIC_WIN_RATE,
        help=f"Minimum win rate vs heuristic to qualify (default: {DEFAULT_MIN_HEURISTIC_WIN_RATE:.0%})",
    )
    parser.add_argument(
        "--no-gating",
        action="store_true",
        help="Disable baseline gating (use original Elo-only selection)",
    )

    args = parser.parse_args()

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }

    # If --no-gating, set thresholds to 0
    min_random = 0.0 if args.no_gating else args.min_random_win_rate
    min_heuristic = 0.0 if args.no_gating else args.min_heuristic_win_rate

    best_ckpt, results = select_best_checkpoint(
        candidate_id=args.candidate_id,
        models_dir=args.models_dir,
        games_per_opponent=args.games,
        board_type=board_type_map[args.board_type],
        num_players=args.num_players,
        min_random_win_rate=min_random,
        min_heuristic_win_rate=min_heuristic,
    )

    if best_ckpt:
        print(f"\n{'='*60}")
        print(f"Best checkpoint: {best_ckpt.name}")

        # Find result for best
        for r in results:
            if r["checkpoint"] == str(best_ckpt):
                print(f"Estimated Elo: {r['estimated_elo']:.0f}")
                break

        if args.copy_best:
            best_path = Path(args.models_dir) / f"{args.candidate_id}_elo_best.pth"
            shutil.copy2(best_ckpt, best_path)
            print(f"Copied to: {best_path}")
    else:
        print("No valid checkpoints found")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
