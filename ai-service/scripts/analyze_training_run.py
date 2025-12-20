#!/usr/bin/env python3
"""Post-Training Analysis Script for RingRift AI.

Analyzes training runs by parsing log files and extracting key metrics.
Provides summary statistics, loss curves, and recommendations.

Usage:
    python scripts/analyze_training_run.py --log logs/training_sq8_2p_enhanced.log
    python scripts/analyze_training_run.py --log logs/training_*.log --compare
    python scripts/analyze_training_run.py --checkpoint models/sq8_2p_v4_enhanced.pth
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    policy_loss: float | None = None
    value_loss: float | None = None
    learning_rate: float | None = None


@dataclass
class TrainingRunSummary:
    """Summary of a complete training run."""
    log_file: str
    total_epochs: int
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float
    epochs: list[EpochMetrics] = field(default_factory=list)
    early_stopped: bool = False
    model_version: str | None = None
    board_type: str | None = None
    training_samples: int | None = None
    validation_samples: int | None = None


def parse_training_log(log_path: Path) -> TrainingRunSummary:
    """Parse a training log file and extract metrics."""
    epochs: list[EpochMetrics] = []
    early_stopped = False
    model_version = None
    board_type = None
    train_samples = None
    val_samples = None

    # Patterns to match
    epoch_pattern = re.compile(
        r"Epoch \[(\d+)/(\d+)\], Train Loss: ([\d.]+), Val Loss: ([\d.]+)"
    )
    batch_pattern = re.compile(
        r"Epoch (\d+), Batch \d+: Loss=([\d.]+) \(Val=([\d.]+), Pol=([\d.]+)\)"
    )
    lr_pattern = re.compile(r"Current LR: ([\d.]+)")
    early_stop_pattern = re.compile(r"Early stopping triggered")
    version_pattern = re.compile(r"Version: (v[\d.]+)")
    board_pattern = re.compile(r"board_type=(\w+)")
    samples_pattern = re.compile(r"Train size: (\d+), Val size: (\d+)")

    current_lr = None
    current_value_loss = None
    current_policy_loss = None

    with open(log_path) as f:
        for line in f:
            # Extract learning rate
            lr_match = lr_pattern.search(line)
            if lr_match:
                current_lr = float(lr_match.group(1))

            # Extract batch losses (for value/policy breakdown)
            batch_match = batch_pattern.search(line)
            if batch_match:
                current_value_loss = float(batch_match.group(3))
                current_policy_loss = float(batch_match.group(4))

            # Extract epoch summary
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                int(epoch_match.group(2))
                train_loss = float(epoch_match.group(3))
                val_loss = float(epoch_match.group(4))

                epochs.append(EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    policy_loss=current_policy_loss,
                    value_loss=current_value_loss,
                    learning_rate=current_lr,
                ))

            # Check for early stopping
            if early_stop_pattern.search(line):
                early_stopped = True

            # Extract model version
            version_match = version_pattern.search(line)
            if version_match:
                model_version = version_match.group(1)

            # Extract board type
            board_match = board_pattern.search(line)
            if board_match:
                board_type = board_match.group(1)

            # Extract sample counts
            samples_match = samples_pattern.search(line)
            if samples_match:
                train_samples = int(samples_match.group(1))
                val_samples = int(samples_match.group(2))

    if not epochs:
        logger.warning(f"No epochs found in {log_path}")
        return TrainingRunSummary(
            log_file=str(log_path),
            total_epochs=0,
            best_epoch=0,
            best_val_loss=float("inf"),
            final_train_loss=float("inf"),
            final_val_loss=float("inf"),
        )

    # Find best epoch (lowest validation loss)
    best_epoch_metrics = min(epochs, key=lambda e: e.val_loss)

    return TrainingRunSummary(
        log_file=str(log_path),
        total_epochs=len(epochs),
        best_epoch=best_epoch_metrics.epoch,
        best_val_loss=best_epoch_metrics.val_loss,
        final_train_loss=epochs[-1].train_loss,
        final_val_loss=epochs[-1].val_loss,
        epochs=epochs,
        early_stopped=early_stopped,
        model_version=model_version,
        board_type=board_type,
        training_samples=train_samples,
        validation_samples=val_samples,
    )


def print_summary(summary: TrainingRunSummary) -> None:
    """Print a formatted summary of the training run."""
    print("\n" + "=" * 60)
    print(f"TRAINING RUN ANALYSIS: {Path(summary.log_file).name}")
    print("=" * 60)

    print(f"\n{'Configuration':^60}")
    print("-" * 60)
    if summary.board_type:
        print(f"  Board Type:     {summary.board_type}")
    if summary.model_version:
        print(f"  Model Version:  {summary.model_version}")
    if summary.training_samples:
        print(f"  Train Samples:  {summary.training_samples}")
    if summary.validation_samples:
        print(f"  Val Samples:    {summary.validation_samples}")

    print(f"\n{'Training Progress':^60}")
    print("-" * 60)
    print(f"  Total Epochs:   {summary.total_epochs}")
    print(f"  Best Epoch:     {summary.best_epoch}")
    print(f"  Early Stopped:  {'Yes' if summary.early_stopped else 'No'}")

    print(f"\n{'Loss Metrics':^60}")
    print("-" * 60)
    print(f"  Best Val Loss:   {summary.best_val_loss:.4f}")
    print(f"  Final Train:     {summary.final_train_loss:.4f}")
    print(f"  Final Val:       {summary.final_val_loss:.4f}")

    # Overfitting indicator
    if summary.epochs:
        train_val_gap = summary.final_val_loss - summary.final_train_loss
        if train_val_gap > summary.final_train_loss * 0.5:
            print(f"  Overfitting:     LIKELY (gap: {train_val_gap:.4f})")
        else:
            print(f"  Overfitting:     Minimal (gap: {train_val_gap:.4f})")

    # Loss trend analysis
    if len(summary.epochs) >= 5:
        last_5_val = [e.val_loss for e in summary.epochs[-5:]]
        trend = last_5_val[-1] - last_5_val[0]
        if trend > 0:
            print(f"  Recent Trend:    Increasing (bad) +{trend:.4f}")
        else:
            print(f"  Recent Trend:    Decreasing (good) {trend:.4f}")

    print(f"\n{'Recommendations':^60}")
    print("-" * 60)

    recommendations = []

    # Early stopping
    if not summary.early_stopped and summary.total_epochs > 0:
        if summary.best_epoch == summary.total_epochs:
            recommendations.append("Model still improving - consider training longer")
        elif summary.best_epoch < summary.total_epochs * 0.5:
            recommendations.append("Best epoch was early - reduce epochs or use early stopping")

    # Overfitting check
    if summary.epochs and summary.final_val_loss > summary.best_val_loss * 1.2:
        recommendations.append("Significant val loss degradation - use checkpoint from best epoch")

    # Sample size
    if summary.training_samples and summary.training_samples < 1000:
        recommendations.append(f"Only {summary.training_samples} samples - consider more data")

    if not recommendations:
        recommendations.append("Training looks healthy!")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print()


def print_loss_curve(summary: TrainingRunSummary, width: int = 50) -> None:
    """Print an ASCII loss curve."""
    if not summary.epochs:
        return

    print(f"{'Loss Curve (Val Loss)':^60}")
    print("-" * 60)

    val_losses = [e.val_loss for e in summary.epochs]
    min_loss = min(val_losses)
    max_loss = max(val_losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1

    for _i, epoch in enumerate(summary.epochs):
        normalized = int((epoch.val_loss - min_loss) / loss_range * width)
        bar = "#" * (width - normalized) + "." * normalized
        marker = "*" if epoch.epoch == summary.best_epoch else " "
        print(f"  {epoch.epoch:3d} |{bar}|{marker} {epoch.val_loss:.2f}")

    print(f"      {'':>{width}} (lower is better)")
    print()


def compare_runs(summaries: list[TrainingRunSummary]) -> None:
    """Print a comparison table of multiple training runs."""
    if not summaries:
        return

    print("\n" + "=" * 80)
    print(f"{'TRAINING RUN COMPARISON':^80}")
    print("=" * 80)

    # Header
    print(f"\n{'Run':<30} {'Epochs':<8} {'Best':<8} {'BestLoss':<12} {'FinalLoss':<12}")
    print("-" * 80)

    # Sort by best val loss
    sorted_summaries = sorted(summaries, key=lambda s: s.best_val_loss)

    for summary in sorted_summaries:
        name = Path(summary.log_file).stem[:28]
        early = "ES" if summary.early_stopped else ""
        print(
            f"{name:<30} {summary.total_epochs:<8} "
            f"{summary.best_epoch:<8} {summary.best_val_loss:<12.4f} "
            f"{summary.final_val_loss:<12.4f} {early}"
        )

    print()

    # Best run
    best = sorted_summaries[0]
    print(f"Best run: {Path(best.log_file).name} (val_loss={best.best_val_loss:.4f})")


def analyze_checkpoint(checkpoint_path: Path) -> None:
    """Analyze a saved checkpoint file."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch required for checkpoint analysis")
        return

    print("\n" + "=" * 60)
    print(f"CHECKPOINT ANALYSIS: {checkpoint_path.name}")
    print("=" * 60)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    print(f"\n{'Checkpoint Contents':^60}")
    print("-" * 60)

    for key in checkpoint:
        if key == "model_state_dict":
            param_count = sum(
                p.numel() for p in checkpoint[key].values()
            )
            print(f"  {key}: {len(checkpoint[key])} tensors ({param_count:,} params)")
        elif key == "optimizer_state_dict":
            print(f"  {key}: present")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: {len(checkpoint[key])} items")
        else:
            print(f"  {key}: {checkpoint[key]}")

    # Version info
    if "version" in checkpoint:
        print(f"\n{'Version Info':^60}")
        print("-" * 60)
        version_info = checkpoint["version"]
        if isinstance(version_info, dict):
            for k, v in version_info.items():
                print(f"  {k}: {v}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze RingRift AI training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --log logs/training_sq8_2p.log
  %(prog)s --log "logs/training_*.log" --compare
  %(prog)s --checkpoint models/sq8_2p_v4.pth
  %(prog)s --log logs/training_sq8_2p.log --curve
        """,
    )
    parser.add_argument(
        "--log",
        type=str,
        help="Training log file(s) to analyze (supports glob patterns)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file to analyze",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple log files",
    )
    parser.add_argument(
        "--curve",
        action="store_true",
        help="Show ASCII loss curve",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    if not args.log and not args.checkpoint:
        parser.print_help()
        return 1

    # Analyze checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1
        analyze_checkpoint(checkpoint_path)

    # Analyze log file(s)
    if args.log:
        log_files = glob.glob(args.log)
        if not log_files:
            logger.error(f"No log files found matching: {args.log}")
            return 1

        summaries = []
        for log_file in sorted(log_files):
            summary = parse_training_log(Path(log_file))
            summaries.append(summary)

        if args.json:
            # Output as JSON
            output = []
            for s in summaries:
                output.append({
                    "log_file": s.log_file,
                    "total_epochs": s.total_epochs,
                    "best_epoch": s.best_epoch,
                    "best_val_loss": s.best_val_loss,
                    "final_train_loss": s.final_train_loss,
                    "final_val_loss": s.final_val_loss,
                    "early_stopped": s.early_stopped,
                    "model_version": s.model_version,
                    "board_type": s.board_type,
                })
            print(json.dumps(output, indent=2))
        elif args.compare and len(summaries) > 1:
            compare_runs(summaries)
        else:
            for summary in summaries:
                print_summary(summary)
                if args.curve:
                    print_loss_curve(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
