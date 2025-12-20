"""NNUE Quality Metrics for RingRift Training Pipeline.

This module provides quality metrics for evaluating NNUE model performance
beyond simple loss values. These metrics help diagnose model weaknesses
and training data issues.

Key metrics:
- Phase-specific loss: Separate loss for early/mid/late game positions
- Win prediction accuracy: How often value sign matches actual outcome
- Calibration: Correlation between predicted values and actual win rates
- Move distribution: Statistics on training data phase coverage

Usage:
    from app.training.nnue_quality_metrics import NNUEQualityMetrics

    metrics = NNUEQualityMetrics()

    # During training
    for features, values, move_numbers in dataloader:
        preds = model(features)
        metrics.update(preds, values, move_numbers)

    # After epoch
    report = metrics.compute()
    print(report)
    metrics.reset()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Configuration for game phase boundaries."""
    early_end: int = 40  # Moves 0-39 are early game
    mid_end: int = 80    # Moves 40-79 are mid game
    # Moves 80+ are late game


@dataclass
class NNUEMetricsReport:
    """Report containing all NNUE quality metrics."""
    # Overall metrics
    total_samples: int = 0
    overall_loss: float = 0.0
    overall_accuracy: float = 0.0  # % where sign(pred) == sign(target)

    # Phase-specific metrics
    early_loss: float = 0.0
    early_accuracy: float = 0.0
    early_count: int = 0

    mid_loss: float = 0.0
    mid_accuracy: float = 0.0
    mid_count: int = 0

    late_loss: float = 0.0
    late_accuracy: float = 0.0
    late_count: int = 0

    # Calibration metrics
    calibration_error: float = 0.0  # Mean absolute difference between pred and actual win rate

    # Distribution stats
    phase_balance: dict[str, float] = field(default_factory=dict)
    value_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "total_samples": self.total_samples,
            "overall/loss": self.overall_loss,
            "overall/accuracy": self.overall_accuracy,
            "early/loss": self.early_loss,
            "early/accuracy": self.early_accuracy,
            "early/count": self.early_count,
            "mid/loss": self.mid_loss,
            "mid/accuracy": self.mid_accuracy,
            "mid/count": self.mid_count,
            "late/loss": self.late_loss,
            "late/accuracy": self.late_accuracy,
            "late/count": self.late_count,
            "calibration_error": self.calibration_error,
            "phase_balance": self.phase_balance,
        }

    def summary_string(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "NNUE Quality Metrics Report",
            "=" * 60,
            f"Total samples: {self.total_samples:,}",
            f"Overall loss: {self.overall_loss:.4f}",
            f"Overall accuracy: {self.overall_accuracy:.1%}",
            "",
            "Phase Breakdown:",
            f"  Early (0-39):  loss={self.early_loss:.4f}, acc={self.early_accuracy:.1%}, n={self.early_count:,} ({self.phase_balance.get('early', 0):.1%})",
            f"  Mid (40-79):   loss={self.mid_loss:.4f}, acc={self.mid_accuracy:.1%}, n={self.mid_count:,} ({self.phase_balance.get('mid', 0):.1%})",
            f"  Late (80+):    loss={self.late_loss:.4f}, acc={self.late_accuracy:.1%}, n={self.late_count:,} ({self.phase_balance.get('late', 0):.1%})",
            "",
            f"Calibration error: {self.calibration_error:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class NNUEQualityMetrics:
    """Tracks quality metrics during NNUE training.

    This class accumulates predictions and targets during training,
    then computes comprehensive quality metrics at the end of each epoch.
    """

    def __init__(self, phase_config: PhaseConfig | None = None):
        """Initialize metrics tracker.

        Args:
            phase_config: Configuration for game phase boundaries
        """
        self.phase_config = phase_config or PhaseConfig()
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        # Accumulated data
        self._predictions: list[float] = []
        self._targets: list[float] = []
        self._move_numbers: list[int] = []
        self._losses: list[float] = []

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        move_numbers: np.ndarray | None = None,
        losses: np.ndarray | None = None,
    ) -> None:
        """Update metrics with a batch of predictions.

        Args:
            predictions: Model predictions (value estimates)
            targets: Ground truth values (+1 win, -1 loss, 0 draw)
            move_numbers: Move numbers for each sample (for phase analysis)
            losses: Per-sample losses (optional, will compute MSE if not provided)
        """
        # Flatten if needed
        preds = np.atleast_1d(predictions.flatten())
        targs = np.atleast_1d(targets.flatten())

        self._predictions.extend(preds.tolist())
        self._targets.extend(targs.tolist())

        # Track move numbers if provided
        if move_numbers is not None:
            moves = np.atleast_1d(move_numbers.flatten())
            self._move_numbers.extend(moves.tolist())
        else:
            # Default to late game if not provided
            self._move_numbers.extend([100] * len(preds))

        # Track losses
        if losses is not None:
            batch_losses = np.atleast_1d(losses.flatten())
            self._losses.extend(batch_losses.tolist())
        else:
            # Compute MSE
            batch_losses = (preds - targs) ** 2
            self._losses.extend(batch_losses.tolist())

    def update_from_torch(
        self,
        predictions,  # torch.Tensor
        targets,  # torch.Tensor
        move_numbers=None,  # Optional torch.Tensor
        losses=None,  # Optional torch.Tensor
    ) -> None:
        """Update metrics from PyTorch tensors.

        Convenience method that handles detach/cpu/numpy conversion.
        """
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()

        moves = None
        if move_numbers is not None:
            moves = move_numbers.detach().cpu().numpy()

        loss_arr = None
        if losses is not None:
            loss_arr = losses.detach().cpu().numpy()

        self.update(preds, targs, moves, loss_arr)

    def compute(self) -> NNUEMetricsReport:
        """Compute all quality metrics from accumulated data.

        Returns:
            NNUEMetricsReport with all computed metrics
        """
        if not self._predictions:
            return NNUEMetricsReport()

        preds = np.array(self._predictions)
        targets = np.array(self._targets)
        moves = np.array(self._move_numbers)
        losses = np.array(self._losses)

        report = NNUEMetricsReport()
        report.total_samples = len(preds)

        # Overall metrics
        report.overall_loss = float(np.mean(losses))
        report.overall_accuracy = self._compute_sign_accuracy(preds, targets)

        # Phase-specific metrics
        early_mask = moves < self.phase_config.early_end
        mid_mask = (moves >= self.phase_config.early_end) & (moves < self.phase_config.mid_end)
        late_mask = moves >= self.phase_config.mid_end

        # Early phase
        if np.any(early_mask):
            report.early_loss = float(np.mean(losses[early_mask]))
            report.early_accuracy = self._compute_sign_accuracy(
                preds[early_mask], targets[early_mask]
            )
            report.early_count = int(np.sum(early_mask))

        # Mid phase
        if np.any(mid_mask):
            report.mid_loss = float(np.mean(losses[mid_mask]))
            report.mid_accuracy = self._compute_sign_accuracy(
                preds[mid_mask], targets[mid_mask]
            )
            report.mid_count = int(np.sum(mid_mask))

        # Late phase
        if np.any(late_mask):
            report.late_loss = float(np.mean(losses[late_mask]))
            report.late_accuracy = self._compute_sign_accuracy(
                preds[late_mask], targets[late_mask]
            )
            report.late_count = int(np.sum(late_mask))

        # Phase balance
        total = len(preds)
        report.phase_balance = {
            "early": report.early_count / total if total > 0 else 0.0,
            "mid": report.mid_count / total if total > 0 else 0.0,
            "late": report.late_count / total if total > 0 else 0.0,
        }

        # Calibration error
        report.calibration_error = self._compute_calibration_error(preds, targets)

        # Value histogram
        report.value_histogram = self._compute_value_histogram(targets)

        return report

    def _compute_sign_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute accuracy of predicting win/loss direction.

        Returns fraction where sign(pred) == sign(target).
        Draws (target=0) are counted as correct if |pred| < 0.5.
        """
        if len(predictions) == 0:
            return 0.0

        correct = 0
        for pred, target in zip(predictions, targets, strict=False):
            if target > 0:
                # Win - prediction should be positive
                if pred > 0:
                    correct += 1
            elif target < 0:
                # Loss - prediction should be negative
                if pred < 0:
                    correct += 1
            else:
                # Draw - prediction should be near zero
                if abs(pred) < 0.5:
                    correct += 1

        return correct / len(predictions)

    def _compute_calibration_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute calibration error using binning.

        Bins predictions by value and compares mean prediction in each bin
        to actual win rate in that bin.
        """
        if len(predictions) < n_bins:
            return 0.0

        # Normalize predictions to [0, 1] range (0.5 = draw)
        norm_preds = (predictions + 1) / 2  # [-1, 1] -> [0, 1]
        norm_targets = (targets + 1) / 2

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        errors = []

        for i in range(n_bins):
            mask = (norm_preds >= bin_edges[i]) & (norm_preds < bin_edges[i + 1])
            if np.sum(mask) > 0:
                mean_pred = np.mean(norm_preds[mask])
                mean_actual = np.mean(norm_targets[mask])
                errors.append(abs(mean_pred - mean_actual))

        return float(np.mean(errors)) if errors else 0.0

    def _compute_value_histogram(
        self,
        targets: np.ndarray,
    ) -> dict[str, int]:
        """Compute histogram of target values."""
        wins = int(np.sum(targets > 0))
        losses = int(np.sum(targets < 0))
        draws = int(np.sum(targets == 0))
        return {"wins": wins, "losses": losses, "draws": draws}


def log_nnue_metrics(
    metrics_logger,  # MetricsLogger instance
    report: NNUEMetricsReport,
    step: int,
    prefix: str = "nnue",
) -> None:
    """Log NNUE quality metrics to the training logger.

    Args:
        metrics_logger: MetricsLogger instance
        report: Computed metrics report
        step: Training step/epoch
        prefix: Prefix for metric names
    """
    # Log scalars
    metrics_logger.log_scalar(f"{prefix}/overall_loss", report.overall_loss, step)
    metrics_logger.log_scalar(f"{prefix}/overall_accuracy", report.overall_accuracy, step)

    # Phase-specific
    metrics_logger.log_scalars(f"{prefix}/loss_by_phase", {
        "early": report.early_loss,
        "mid": report.mid_loss,
        "late": report.late_loss,
    }, step)

    metrics_logger.log_scalars(f"{prefix}/accuracy_by_phase", {
        "early": report.early_accuracy,
        "mid": report.mid_accuracy,
        "late": report.late_accuracy,
    }, step)

    # Data balance
    metrics_logger.log_scalars(f"{prefix}/phase_balance", report.phase_balance, step)

    # Calibration
    metrics_logger.log_scalar(f"{prefix}/calibration_error", report.calibration_error, step)


def analyze_dataset_quality(
    db_path: str,
    board_type: str = "square8",
    num_players: int = 2,
) -> dict[str, Any]:
    """Analyze the quality and balance of an NNUE training dataset.

    Args:
        db_path: Path to SQLite game database
        board_type: Board type string
        num_players: Number of players

    Returns:
        Dictionary with dataset statistics
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count snapshots by move number range
    cursor.execute("""
        SELECT
            SUM(CASE WHEN s.move_number < 40 THEN 1 ELSE 0 END) as early,
            SUM(CASE WHEN s.move_number >= 40 AND s.move_number < 80 THEN 1 ELSE 0 END) as mid,
            SUM(CASE WHEN s.move_number >= 80 THEN 1 ELSE 0 END) as late,
            COUNT(*) as total
        FROM games g
        JOIN game_state_snapshots s ON g.game_id = s.game_id
        WHERE g.game_status = 'completed'
          AND g.winner IS NOT NULL
          AND g.board_type = ?
          AND g.num_players = ?
    """, (board_type.lower(), num_players))

    row = cursor.fetchone()
    early, mid, late, total = row if row else (0, 0, 0, 0)

    # Count outcomes
    cursor.execute("""
        SELECT winner, COUNT(*)
        FROM games
        WHERE game_status = 'completed'
          AND board_type = ?
          AND num_players = ?
        GROUP BY winner
    """, (board_type.lower(), num_players))

    winner_dist = {str(r[0]): r[1] for r in cursor.fetchall()}

    # Average game length
    cursor.execute("""
        SELECT AVG(total_moves), MIN(total_moves), MAX(total_moves)
        FROM games
        WHERE game_status = 'completed'
          AND board_type = ?
          AND num_players = ?
    """, (board_type.lower(), num_players))

    avg_moves, min_moves, max_moves = cursor.fetchone()

    conn.close()

    return {
        "total_snapshots": total,
        "phase_distribution": {
            "early": early,
            "mid": mid,
            "late": late,
        },
        "phase_percentages": {
            "early": early / total * 100 if total > 0 else 0,
            "mid": mid / total * 100 if total > 0 else 0,
            "late": late / total * 100 if total > 0 else 0,
        },
        "winner_distribution": winner_dist,
        "game_length": {
            "avg": avg_moves,
            "min": min_moves,
            "max": max_moves,
        },
        "quality_warnings": _generate_quality_warnings(early, mid, late, total),
    }


def _generate_quality_warnings(
    early: int,
    mid: int,
    late: int,
    total: int,
) -> list[str]:
    """Generate warnings about dataset quality issues."""
    warnings = []

    if total == 0:
        return ["No training samples found"]

    early_pct = early / total * 100
    mid_pct = mid / total * 100
    late_pct = late / total * 100

    # Check phase balance
    if early_pct < 10:
        warnings.append(f"Early game underrepresented ({early_pct:.1f}% < 10%)")
    if mid_pct < 20:
        warnings.append(f"Mid game underrepresented ({mid_pct:.1f}% < 20%)")
    if late_pct > 70:
        warnings.append(f"Late game overrepresented ({late_pct:.1f}% > 70%)")

    # Check absolute counts
    if early < 1000:
        warnings.append(f"Very few early game samples ({early:,} < 1000)")
    if mid < 5000:
        warnings.append(f"Few mid game samples ({mid:,} < 5000)")

    return warnings


if __name__ == "__main__":
    # Test the module
    import argparse

    parser = argparse.ArgumentParser(description="Analyze NNUE training data quality")
    parser.add_argument("db_path", help="Path to game database")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    args = parser.parse_args()

    analysis = analyze_dataset_quality(args.db_path, args.board, args.players)

    print("\n" + "=" * 60)
    print("NNUE Training Data Quality Analysis")
    print("=" * 60)
    print(f"\nTotal snapshots: {analysis['total_snapshots']:,}")
    print("\nPhase Distribution:")
    for phase, count in analysis['phase_distribution'].items():
        pct = analysis['phase_percentages'][phase]
        print(f"  {phase.capitalize()}: {count:,} ({pct:.1f}%)")

    print(f"\nGame Length: avg={analysis['game_length']['avg']:.1f}, "
          f"min={analysis['game_length']['min']}, max={analysis['game_length']['max']}")

    print("\nWinner Distribution:", analysis['winner_distribution'])

    if analysis['quality_warnings']:
        print("\nWarnings:")
        for w in analysis['quality_warnings']:
            print(f"  - {w}")
    else:
        print("\nNo quality warnings.")
    print("=" * 60)
