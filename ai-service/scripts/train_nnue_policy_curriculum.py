#!/usr/bin/env python
"""Curriculum training for NNUE policy models.

Implements staged training that progresses from late-game (simpler decisions)
to opening (more strategic complexity). Each stage uses transfer learning
from the previous stage's weights.

Stages (in training order):
1. Endgame (move 150+) - 20 epochs: Clear winning/losing moves
2. Late-mid (move 80-150) - 20 epochs: Strategic decisions
3. Midgame (move 30-80) - 25 epochs: Tactical complexity
4. Opening (move 0-30) - 30 epochs: Opening theory
5. Full (all moves) - 25 epochs: Polish with complete data

Usage:
    python scripts/train_nnue_policy_curriculum.py \\
        --db data/games/*.db \\
        --board-type hexagonal \\
        --num-players 2

    # Skip specific stages
    python scripts/train_nnue_policy_curriculum.py \\
        --db data/games/*.db \\
        --board-type square8 \\
        --skip-stages endgame late-mid

    # Custom epochs per stage
    python scripts/train_nnue_policy_curriculum.py \\
        --db data/games/*.db \\
        --board-type square8 \\
        --endgame-epochs 30 \\
        --full-epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    min_move: int
    max_move: int
    epochs: int
    description: str


DEFAULT_STAGES = [
    CurriculumStage("endgame", 150, 999999, 20, "Clear winning/losing moves"),
    CurriculumStage("late-mid", 80, 149, 20, "Strategic decisions"),
    CurriculumStage("midgame", 30, 79, 25, "Tactical complexity"),
    CurriculumStage("opening", 0, 29, 30, "Opening theory"),
    CurriculumStage("full", 0, 999999, 25, "Polish with all data"),
]


def run_training_stage(
    stage: CurriculumStage,
    db_paths: List[str],
    board_type: str,
    num_players: int,
    output_dir: Path,
    pretrained_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run a single curriculum stage.

    Returns:
        Tuple of (model_path, training_report)
    """
    stage_dir = output_dir / f"stage_{stage.name}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    model_path = stage_dir / f"nnue_policy_{board_type}_{num_players}p.pt"

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_nnue_policy.py"),
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--epochs", str(stage.epochs),
        "--min-move-number", str(stage.min_move),
        "--max-move-number", str(stage.max_move),
        "--run-dir", str(stage_dir),
    ]

    # Add db paths
    for db in db_paths:
        cmd.extend(["--db", db])

    # Add pretrained if provided (transfer learning)
    if pretrained_path:
        cmd.extend(["--pretrained", pretrained_path])

    # Add extra args
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running stage '{stage.name}': moves {stage.min_move}-{stage.max_move}, {stage.epochs} epochs")
    logger.info(f"  {stage.description}")
    if pretrained_path:
        logger.info(f"  Transfer learning from: {pretrained_path}")

    # Run training
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(f"Stage '{stage.name}' failed with return code {result.returncode}")
        return str(model_path), {"error": f"Training failed with code {result.returncode}"}

    # Load report if exists (try both naming conventions)
    report = {}
    for report_name in ["nnue_policy_training_report.json", "training_report.json"]:
        report_path = stage_dir / report_name
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            break

    return str(model_path), report


def run_curriculum_training(
    db_paths: List[str],
    board_type: str,
    num_players: int,
    output_dir: Path,
    stages: List[CurriculumStage],
    skip_stages: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full curriculum training.

    Returns:
        Summary report of all stages
    """
    skip_stages = skip_stages or []

    logger.info("=" * 60)
    logger.info("CURRICULUM TRAINING")
    logger.info("=" * 60)
    logger.info(f"Board type: {board_type}")
    logger.info(f"Num players: {num_players}")
    logger.info(f"Stages: {len(stages)} (skipping: {skip_stages or 'none'})")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    current_model = None

    for i, stage in enumerate(stages):
        if stage.name in skip_stages:
            logger.info(f"Skipping stage '{stage.name}'")
            continue

        logger.info("")
        logger.info(f"[{i+1}/{len(stages)}] Stage: {stage.name.upper()}")
        logger.info("-" * 40)

        model_path, report = run_training_stage(
            stage=stage,
            db_paths=db_paths,
            board_type=board_type,
            num_players=num_players,
            output_dir=output_dir,
            pretrained_path=current_model,
            extra_args=extra_args,
        )

        if "error" not in report:
            current_model = model_path

        results.append({
            "stage": stage.name,
            "model_path": model_path,
            "epochs": stage.epochs,
            "move_range": [stage.min_move, stage.max_move],
            "report": report,
        })

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CURRICULUM COMPLETE")
    logger.info("=" * 60)

    for r in results:
        acc = r["report"].get("final_val_policy_accuracy", "N/A")
        if isinstance(acc, float):
            acc = f"{acc:.2%}"
        logger.info(f"  {r['stage']}: accuracy={acc}")

    if current_model:
        logger.info("")
        logger.info(f"Final model: {current_model}")

    # Save summary
    summary = {
        "board_type": board_type,
        "num_players": num_players,
        "timestamp": datetime.now().isoformat(),
        "stages": results,
        "final_model": current_model,
    }

    summary_path = output_dir / "curriculum_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum training for NNUE policy models"
    )

    # Required
    parser.add_argument(
        "--db",
        type=str,
        nargs="+",
        required=True,
        help="Paths to game database files (supports glob)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        required=True,
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Board type",
    )

    # Optional
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: curriculum_runs/<board>_<players>p)",
    )

    # Stage control
    parser.add_argument(
        "--skip-stages",
        type=str,
        nargs="*",
        default=[],
        help="Stages to skip (endgame, late-mid, midgame, opening, full)",
    )

    # Custom epochs
    parser.add_argument("--endgame-epochs", type=int, default=20, help="Epochs for endgame stage")
    parser.add_argument("--late-mid-epochs", type=int, default=20, help="Epochs for late-mid stage")
    parser.add_argument("--midgame-epochs", type=int, default=25, help="Epochs for midgame stage")
    parser.add_argument("--opening-epochs", type=int, default=30, help="Epochs for opening stage")
    parser.add_argument("--full-epochs", type=int, default=25, help="Epochs for full stage")

    # Training parameters passed to train_nnue_policy.py
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension (auto if None)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per stage")
    parser.add_argument("--num-workers", type=int, default=0, help="Extraction workers (0=auto)")

    args = parser.parse_args()

    # Expand db paths
    import glob
    db_paths = []
    for pattern in args.db:
        expanded = glob.glob(pattern)
        if expanded:
            db_paths.extend(expanded)
        else:
            db_paths.append(pattern)

    if not db_paths:
        logger.error("No database files found")
        return 1

    logger.info(f"Found {len(db_paths)} database files")

    # Build stages with custom epochs
    stages = [
        CurriculumStage("endgame", 150, 999999, args.endgame_epochs, "Clear winning/losing moves"),
        CurriculumStage("late-mid", 80, 149, args.late_mid_epochs, "Strategic decisions"),
        CurriculumStage("midgame", 30, 79, args.midgame_epochs, "Tactical complexity"),
        CurriculumStage("opening", 0, 29, args.opening_epochs, "Opening theory"),
        CurriculumStage("full", 0, 999999, args.full_epochs, "Polish with all data"),
    ]

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("curriculum_runs") / f"{args.board_type}_{args.num_players}p"

    # Build extra args to pass through
    extra_args = [
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
    ]
    if args.hidden_dim is not None:
        extra_args.extend(["--hidden-dim", str(args.hidden_dim)])
    if args.max_samples is not None:
        extra_args.extend(["--max-samples", str(args.max_samples)])
    if args.num_workers != 0:
        extra_args.extend(["--num-workers", str(args.num_workers)])

    # Run curriculum
    run_curriculum_training(
        db_paths=db_paths,
        board_type=args.board_type,
        num_players=args.num_players,
        output_dir=output_dir,
        stages=stages,
        skip_stages=args.skip_stages,
        extra_args=extra_args,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
