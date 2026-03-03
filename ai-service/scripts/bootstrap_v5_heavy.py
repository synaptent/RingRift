#!/usr/bin/env python3
"""Bootstrap a v5-heavy model for a single board configuration.

This script runs the full pipeline to bootstrap a v5-heavy model from an
existing canonical v2 model checkpoint:

1. Export training data with full heuristic features (49-feature mode)
2. Transfer compatible weights from the canonical v2 model to a new v5-heavy model
3. Train the v5-heavy model on the exported data
4. Evaluate the trained model via a quick baseline gauntlet

Usage (from ai-service/):
    # Full pipeline for hex8 2-player
    PYTHONPATH=. python3 scripts/bootstrap_v5_heavy.py \
        --board-type hex8 --num-players 2

    # Skip export if NPZ already exists
    PYTHONPATH=. python3 scripts/bootstrap_v5_heavy.py \
        --board-type hex8 --num-players 2 --skip-export

    # Dry run (preview actions without executing)
    PYTHONPATH=. python3 scripts/bootstrap_v5_heavy.py \
        --board-type square8 --num-players 4 --dry-run

    # Custom epochs, skip evaluation
    PYTHONPATH=. python3 scripts/bootstrap_v5_heavy.py \
        --board-type hexagonal --num-players 2 --epochs 50 --skip-eval
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve ai-service root relative to this script
AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _config_key(board_type: str, num_players: int) -> str:
    """Return the canonical config key, e.g. 'hex8_2p'."""
    return f"{board_type}_{num_players}p"


def _canonical_model_path(config_key: str) -> Path:
    """Path to the canonical v2 model for this config."""
    return AI_SERVICE_ROOT / "models" / f"canonical_{config_key}.pth"


def _bootstrap_model_path(config_key: str) -> Path:
    """Path to the bootstrapped v5-heavy model."""
    return AI_SERVICE_ROOT / "models" / f"{config_key}_v5heavy_bootstrap.pth"


def _npz_path(config_key: str) -> Path:
    """Path to the v5-heavy training data NPZ."""
    return AI_SERVICE_ROOT / "data" / "training" / f"{config_key}_v5heavy.npz"


def _subprocess_env() -> dict[str, str]:
    """Build environment dict with PYTHONPATH set to ai-service root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AI_SERVICE_ROOT)
    return env


# ---------------------------------------------------------------------------
# Step 1: Export training data with full heuristic features
# ---------------------------------------------------------------------------

def step_export(board_type: str, num_players: int, config_key: str, dry_run: bool) -> Path:
    """Export training data from game databases with --full-heuristics."""
    npz = _npz_path(config_key)
    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "export_replay_dataset.py"),
        "--use-discovery",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--include-heuristics",
        "--full-heuristics",
        "--allow-pending-gate",
        "--allow-noncanonical",
        "--output", str(npz),
    ]
    logger.info("STEP 1: Export training data with full heuristics")
    logger.info("  Command: %s", " ".join(cmd))
    if dry_run:
        logger.info("  [DRY RUN] Skipping export")
        return npz

    npz.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        env=_subprocess_env(),
        cwd=str(AI_SERVICE_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Export failed with exit code {result.returncode}")

    if not npz.exists():
        raise FileNotFoundError(f"Export produced no output file: {npz}")
    size_mb = npz.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        raise RuntimeError(
            f"Export output is suspiciously small ({size_mb:.2f} MB). "
            "Check that game databases have sufficient data."
        )
    logger.info("  Export complete: %s (%.1f MB)", npz, size_mb)
    return npz


# ---------------------------------------------------------------------------
# Step 2: Transfer weights from canonical v2 model to v5-heavy
# ---------------------------------------------------------------------------

def step_transfer_weights(
    board_type: str,
    num_players: int,
    config_key: str,
    dry_run: bool,
) -> tuple[Path, dict[str, list[str]]]:
    """Create v5-heavy model and transfer compatible weights from canonical v2.

    Returns:
        Tuple of (bootstrap_model_path, transfer_result_dict)
    """
    logger.info("STEP 2: Transfer weights from canonical v2 to v5-heavy")
    canonical = _canonical_model_path(config_key)
    bootstrap = _bootstrap_model_path(config_key)

    if not canonical.exists():
        logger.warning(
            "  Canonical v2 model not found: %s. "
            "v5-heavy model will be initialized from scratch.",
            canonical,
        )
        if dry_run:
            logger.info("  [DRY RUN] Would create random v5-heavy model at %s", bootstrap)
            return bootstrap, {"missing_keys": [], "unexpected_keys": []}

        # Create a fresh v5-heavy model and save it
        model = _create_v5_heavy_model(board_type, num_players)
        _save_bootstrap_checkpoint(model, bootstrap, board_type, num_players, transfer_from=None)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("  Created fresh v5-heavy model: %s (%d parameters)", bootstrap, param_count)
        return bootstrap, {"missing_keys": ["(all layers — random init)"], "unexpected_keys": []}

    logger.info("  Source canonical model: %s", canonical)
    logger.info("  Target bootstrap model: %s", bootstrap)

    if dry_run:
        logger.info("  [DRY RUN] Would transfer compatible layers from v2 -> v5-heavy")
        return bootstrap, {"missing_keys": [], "unexpected_keys": []}

    # Create v5-heavy model
    model = _create_v5_heavy_model(board_type, num_players)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("  Created v5-heavy model: %d parameters", param_count)

    # Transfer compatible weights using load_weights_only(strict=False)
    # Suppress the deprecation warning from checkpointing module
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from app.training.checkpointing import load_weights_only

    result = load_weights_only(
        path=str(canonical),
        model=model,
        strict=False,
    )

    # Log transfer results
    total_layers = len(dict(model.named_parameters()))
    transferred = total_layers - len(result["missing_keys"])
    logger.info("  Layers transferred:      %d / %d", transferred, total_layers)
    if result["missing_keys"]:
        logger.info("  Layers randomly initialized (%d):", len(result["missing_keys"]))
        for key in result["missing_keys"]:
            logger.info("    - %s", key)
    if result["unexpected_keys"]:
        logger.info("  Layers in checkpoint but not in v5-heavy (%d):", len(result["unexpected_keys"]))
        for key in result["unexpected_keys"]:
            logger.info("    - %s", key)

    # Save the bootstrap checkpoint
    _save_bootstrap_checkpoint(model, bootstrap, board_type, num_players, transfer_from=str(canonical))
    logger.info("  Saved bootstrap model: %s", bootstrap)

    return bootstrap, result


def _create_v5_heavy_model(board_type: str, num_players: int):
    """Create a v5-heavy model using the factory function."""
    from app.ai.neural_net.v5_heavy import (
        create_v5_heavy_model,
        NUM_HEURISTIC_FEATURES_FULL,
    )
    return create_v5_heavy_model(
        board_type=board_type,
        num_players=num_players,
        num_heuristics=NUM_HEURISTIC_FEATURES_FULL,
    )


def _save_bootstrap_checkpoint(
    model,
    path: Path,
    board_type: str,
    num_players: int,
    transfer_from: str | None,
) -> None:
    """Save model as a bootstrap checkpoint with metadata."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "transfer_from": transfer_from,
        "transfer_type": "v2_to_v5heavy",
        "num_players": num_players,
        "_versioning_metadata": {
            "config": {
                "num_players": num_players,
                "board_type": board_type,
                "model_version": "v5-heavy",
                "transfer_learning": True,
            }
        },
    }
    # Atomic save: write to temp, then rename
    temp_path = path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temp_path)
    temp_path.rename(path)


# ---------------------------------------------------------------------------
# Step 3: Train the v5-heavy model
# ---------------------------------------------------------------------------

def step_train(
    board_type: str,
    num_players: int,
    config_key: str,
    npz_path: Path,
    bootstrap_path: Path,
    epochs: int,
    dry_run: bool,
) -> int:
    """Train the v5-heavy model. Returns subprocess exit code."""
    logger.info("STEP 3: Train v5-heavy model")
    cmd = [
        sys.executable,
        "-m", "app.training.train",
        "--model-version", "v5-heavy",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--data-path", str(npz_path),
        "--init-weights", str(bootstrap_path),
        "--epochs", str(epochs),
    ]
    logger.info("  Command: %s", " ".join(cmd))
    if dry_run:
        logger.info("  [DRY RUN] Skipping training")
        return 0

    if not npz_path.exists():
        raise FileNotFoundError(f"Training data not found: {npz_path}")
    if not bootstrap_path.exists():
        raise FileNotFoundError(f"Bootstrap model not found: {bootstrap_path}")

    result = subprocess.run(
        cmd,
        env=_subprocess_env(),
        cwd=str(AI_SERVICE_ROOT),
    )
    if result.returncode != 0:
        logger.error("  Training failed with exit code %d", result.returncode)
    else:
        logger.info("  Training completed successfully")
    return result.returncode


# ---------------------------------------------------------------------------
# Step 4: Evaluate the trained model
# ---------------------------------------------------------------------------

def step_evaluate(
    board_type: str,
    num_players: int,
    config_key: str,
    dry_run: bool,
) -> dict | None:
    """Run a quick baseline gauntlet on the trained model.

    Looks for the trained model at the canonical output location that
    train.py writes to (models/ringrift_best_{config}.pth or the
    bootstrap path).
    """
    logger.info("STEP 4: Evaluate trained model (quick gauntlet)")

    # train.py typically saves the best model as ringrift_best_{config}.pth
    # or the canonical path. Check both.
    candidate_paths = [
        AI_SERVICE_ROOT / "models" / f"ringrift_best_{config_key}.pth",
        AI_SERVICE_ROOT / "models" / f"canonical_{config_key}.pth",
        _bootstrap_model_path(config_key),
    ]
    model_path = None
    for p in candidate_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        logger.warning("  No trained model found for evaluation. Checked: %s",
                       [str(p) for p in candidate_paths])
        return None

    logger.info("  Evaluating model: %s", model_path)
    if dry_run:
        logger.info("  [DRY RUN] Skipping evaluation")
        return None

    try:
        from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

        # Map board_type string to BoardType enum
        from app.rules.board_config import BoardType
        board_type_enum = BoardType[board_type.upper()]

        gauntlet_result = run_baseline_gauntlet(
            model_path=str(model_path),
            board_type=board_type_enum,
            num_players=num_players,
            games_per_opponent=10,
            opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
            parallel_opponents=False,  # Avoid nested thread pool deadlock
            use_search=False,  # Fast evaluation without MCTS
            model_version="v5-heavy",
            verbose=True,
            store_results=False,
        )

        logger.info("  Gauntlet results:")
        logger.info("    Overall passed: %s", gauntlet_result.passed)
        if hasattr(gauntlet_result, "opponent_results"):
            for opp_result in gauntlet_result.opponent_results:
                name = getattr(opp_result, "opponent_name", "unknown")
                wins = getattr(opp_result, "wins", 0)
                games = getattr(opp_result, "games_played", 0)
                rate = wins / games if games > 0 else 0.0
                logger.info("    vs %s: %d/%d (%.1f%%)", name, wins, games, rate * 100)
        return {"passed": gauntlet_result.passed}

    except Exception as e:
        logger.warning("  Gauntlet evaluation failed: %s", e)
        logger.info("  This is non-fatal. The trained model is still available.")
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    config_key: str,
    npz_path: Path,
    bootstrap_path: Path,
    transfer_result: dict[str, list[str]],
    train_exit_code: int,
    eval_result: dict | None,
    elapsed: float,
    dry_run: bool,
) -> None:
    """Print a summary of the bootstrap pipeline run."""
    print()
    print("=" * 70)
    print(f"  V5-HEAVY BOOTSTRAP REPORT: {config_key}")
    print("=" * 70)
    print()

    if dry_run:
        print("  Mode: DRY RUN (no actions taken)")
        print()

    # Export
    if npz_path.exists():
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"  Export:     {npz_path.name} ({size_mb:.1f} MB)")
    else:
        print(f"  Export:     {npz_path.name} (not found)")

    # Transfer
    total_missing = len(transfer_result.get("missing_keys", []))
    total_unexpected = len(transfer_result.get("unexpected_keys", []))
    print(f"  Transfer:   {bootstrap_path.name}")
    print(f"              New layers (random init): {total_missing}")
    print(f"              Skipped (not in v5-heavy): {total_unexpected}")

    # Training
    if train_exit_code == 0:
        print("  Training:   COMPLETED")
    else:
        print(f"  Training:   FAILED (exit code {train_exit_code})")

    # Evaluation
    if eval_result is not None:
        passed = eval_result.get("passed", False)
        status = "PASSED" if passed else "FAILED"
        print(f"  Evaluation: {status}")
    else:
        print("  Evaluation: SKIPPED")

    print()
    print(f"  Elapsed:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap a v5-heavy model for one board configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--board-type",
        required=True,
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type to bootstrap.",
    )
    parser.add_argument(
        "--num-players",
        required=True,
        type=int,
        choices=[2, 3, 4],
        help="Number of players.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the export step (use existing NPZ).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation step.",
    )

    args = parser.parse_args()
    board_type: str = args.board_type
    num_players: int = args.num_players
    epochs: int = args.epochs
    dry_run: bool = args.dry_run
    skip_export: bool = args.skip_export
    skip_eval: bool = args.skip_eval

    config_key = _config_key(board_type, num_players)
    logger.info("Bootstrapping v5-heavy model for %s", config_key)
    if dry_run:
        logger.info("[DRY RUN MODE] No files will be created or modified.")

    start_time = time.monotonic()
    npz_path = _npz_path(config_key)
    train_exit_code = -1
    transfer_result: dict[str, list[str]] = {"missing_keys": [], "unexpected_keys": []}
    eval_result: dict | None = None
    bootstrap_path = _bootstrap_model_path(config_key)

    try:
        # Step 1: Export
        if skip_export:
            logger.info("STEP 1: Export SKIPPED (--skip-export)")
            if not npz_path.exists() and not dry_run:
                raise FileNotFoundError(
                    f"--skip-export specified but NPZ not found: {npz_path}. "
                    "Run without --skip-export first."
                )
        else:
            npz_path = step_export(board_type, num_players, config_key, dry_run)

        # Step 2: Transfer weights
        bootstrap_path, transfer_result = step_transfer_weights(
            board_type, num_players, config_key, dry_run,
        )

        # Step 3: Train
        train_exit_code = step_train(
            board_type, num_players, config_key,
            npz_path, bootstrap_path, epochs, dry_run,
        )

        # Step 4: Evaluate
        if skip_eval:
            logger.info("STEP 4: Evaluation SKIPPED (--skip-eval)")
        elif train_exit_code != 0 and not dry_run:
            logger.info("STEP 4: Evaluation SKIPPED (training failed)")
        else:
            eval_result = step_evaluate(board_type, num_players, config_key, dry_run)

    except Exception:
        logger.exception("Bootstrap pipeline failed")

    elapsed = time.monotonic() - start_time
    print_report(
        config_key, npz_path, bootstrap_path,
        transfer_result, train_exit_code, eval_result,
        elapsed, dry_run,
    )

    return 0 if train_exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
