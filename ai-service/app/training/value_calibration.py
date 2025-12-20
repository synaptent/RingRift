"""
Value Head Calibration for RingRift AI.

Ensures value predictions are well-calibrated to actual game outcomes,
improving MCTS search quality and decision-making.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """A bin for calibration analysis."""
    lower: float
    upper: float
    predictions: List[float] = field(default_factory=list)
    outcomes: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.predictions)

    @property
    def mean_prediction(self) -> float:
        return float(np.mean(self.predictions)) if self.predictions else 0.0

    @property
    def mean_outcome(self) -> float:
        return float(np.mean(self.outcomes)) if self.outcomes else 0.0

    @property
    def calibration_error(self) -> float:
        """Absolute difference between mean prediction and outcome."""
        return abs(self.mean_prediction - self.mean_outcome)


@dataclass
class CalibrationReport:
    """Report of value head calibration analysis."""
    ece: float                           # Expected Calibration Error
    mce: float                           # Maximum Calibration Error
    overconfidence: float                # Degree of overconfidence (-1 to 1)
    bins: List[CalibrationBin] = field(default_factory=list)
    total_samples: int = 0
    reliability_diagram: Optional[Dict[str, List[float]]] = None
    optimal_temperature: Optional[float] = None
    brier_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ece': self.ece,
            'mce': self.mce,
            'overconfidence': self.overconfidence,
            'total_samples': self.total_samples,
            'optimal_temperature': self.optimal_temperature,
            'brier_score': self.brier_score,
            'bins': [
                {
                    'lower': b.lower,
                    'upper': b.upper,
                    'count': b.count,
                    'mean_prediction': b.mean_prediction,
                    'mean_outcome': b.mean_outcome,
                    'calibration_error': b.calibration_error
                }
                for b in self.bins
            ]
        }


class ValueCalibrator:
    """
    Analyzes and calibrates value head predictions.
    """

    def __init__(self, num_bins: int = 10):
        """
        Args:
            num_bins: Number of bins for calibration analysis
        """
        self.num_bins = num_bins
        self.predictions: List[float] = []
        self.outcomes: List[float] = []
        self._temperature = 1.0

    def add_sample(self, prediction: float, outcome: float):
        """
        Add a prediction-outcome pair.

        Args:
            prediction: Value prediction from model (typically -1 to 1)
            outcome: Actual game outcome (-1=loss, 0=draw, 1=win)
        """
        self.predictions.append(prediction)
        self.outcomes.append(outcome)

    def add_batch(self, predictions: List[float], outcomes: List[float]):
        """Add a batch of prediction-outcome pairs."""
        assert len(predictions) == len(outcomes)
        self.predictions.extend(predictions)
        self.outcomes.extend(outcomes)

    def clear(self):
        """Clear all samples."""
        self.predictions = []
        self.outcomes = []

    def compute_calibration(self) -> CalibrationReport:
        """
        Compute calibration metrics.

        Returns:
            CalibrationReport with ECE, MCE, and bin details
        """
        if not self.predictions:
            return CalibrationReport(ece=0.0, mce=0.0, overconfidence=0.0)

        predictions = np.array(self.predictions)
        outcomes = np.array(self.outcomes)

        # Convert from [-1, 1] to [0, 1] for calibration analysis
        pred_probs = (predictions + 1) / 2
        outcome_probs = (outcomes + 1) / 2

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bins = []

        for i in range(self.num_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            mask = (pred_probs >= lower) & (pred_probs < upper)
            if i == self.num_bins - 1:  # Include upper bound in last bin
                mask = (pred_probs >= lower) & (pred_probs <= upper)

            bin_obj = CalibrationBin(
                lower=lower,
                upper=upper,
                predictions=pred_probs[mask].tolist(),
                outcomes=outcome_probs[mask].tolist()
            )
            bins.append(bin_obj)

        # Compute ECE (Expected Calibration Error)
        total = len(predictions)
        ece = sum(bin.count * bin.calibration_error for bin in bins) / total

        # Compute MCE (Maximum Calibration Error)
        mce = max(bin.calibration_error for bin in bins if bin.count > 0)

        # Compute overconfidence
        # Positive = overconfident (predicts more extreme than outcomes)
        # Negative = underconfident
        pred_extremity = np.mean(np.abs(predictions))
        outcome_extremity = np.mean(np.abs(outcomes))
        overconfidence = pred_extremity - outcome_extremity

        # Compute Brier score
        brier_score = np.mean((pred_probs - outcome_probs) ** 2)

        # Create reliability diagram data
        reliability_diagram = {
            'bin_midpoints': [(b.lower + b.upper) / 2 for b in bins],
            'mean_predictions': [b.mean_prediction for b in bins],
            'mean_outcomes': [b.mean_outcome for b in bins],
            'counts': [b.count for b in bins]
        }

        return CalibrationReport(
            ece=ece,
            mce=mce,
            overconfidence=overconfidence,
            bins=bins,
            total_samples=total,
            reliability_diagram=reliability_diagram,
            brier_score=brier_score
        )

    def find_optimal_temperature(
        self,
        temp_range: Tuple[float, float] = (0.5, 2.0),
        num_steps: int = 20
    ) -> float:
        """
        Find optimal temperature scaling parameter.

        Temperature scaling calibrates predictions by dividing logits by T.
        T > 1 makes predictions less confident (flatter)
        T < 1 makes predictions more confident (sharper)

        Returns:
            Optimal temperature value
        """
        if not self.predictions:
            return 1.0

        predictions = np.array(self.predictions)
        outcomes = np.array(self.outcomes)

        best_temp = 1.0
        best_nll = float('inf')

        for temp in np.linspace(temp_range[0], temp_range[1], num_steps):
            # Apply temperature scaling
            scaled_preds = np.tanh(np.arctanh(np.clip(predictions, -0.999, 0.999)) / temp)

            # Convert to probabilities
            pred_probs = (scaled_preds + 1) / 2
            outcome_probs = (outcomes + 1) / 2

            # Compute negative log-likelihood
            eps = 1e-7
            nll = -np.mean(
                outcome_probs * np.log(pred_probs + eps) +
                (1 - outcome_probs) * np.log(1 - pred_probs + eps)
            )

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self._temperature = best_temp
        return best_temp

    def calibrate_prediction(self, prediction: float) -> float:
        """
        Apply temperature scaling to a prediction.

        Args:
            prediction: Raw value prediction (-1 to 1)

        Returns:
            Calibrated prediction
        """
        if self._temperature == 1.0:
            return prediction

        # Apply temperature scaling via arctanh/tanh
        clipped = np.clip(prediction, -0.999, 0.999)
        scaled = np.tanh(np.arctanh(clipped) / self._temperature)
        return float(scaled)

    def calibrate_batch(self, predictions: List[float]) -> List[float]:
        """Apply temperature scaling to a batch of predictions."""
        return [self.calibrate_prediction(p) for p in predictions]

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = max(0.1, min(10.0, value))


class PlattScaler:
    """
    Platt scaling for value calibration.

    Fits a sigmoid function: P(y=1|f) = 1 / (1 + exp(A*f + B))
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(self, predictions: List[float], outcomes: List[float],
            learning_rate: float = 0.01, max_iterations: int = 1000,
            tolerance: float = 1e-6) -> 'PlattScaler':
        """
        Fit Platt scaling parameters using gradient descent.

        Args:
            predictions: Model predictions (-1 to 1)
            outcomes: Actual outcomes (-1, 0, or 1)
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Self for chaining
        """
        # Convert to numpy
        f = np.array(predictions)
        y = (np.array(outcomes) + 1) / 2  # Convert to [0, 1]

        # Initialize parameters
        a = 1.0
        b = 0.0

        for iteration in range(max_iterations):
            # Compute probabilities
            z = a * f + b
            p = 1 / (1 + np.exp(-z))

            # Compute gradients (negative log-likelihood)
            grad_a = np.mean((p - y) * f)
            grad_b = np.mean(p - y)

            # Update parameters
            a_new = a - learning_rate * grad_a
            b_new = b - learning_rate * grad_b

            # Check convergence
            if abs(a_new - a) < tolerance and abs(b_new - b) < tolerance:
                break

            a, b = a_new, b_new

        self.a = a
        self.b = b
        self._fitted = True

        return self

    def calibrate(self, prediction: float) -> float:
        """
        Calibrate a single prediction.

        Args:
            prediction: Raw value prediction (-1 to 1)

        Returns:
            Calibrated prediction (-1 to 1)
        """
        z = self.a * prediction + self.b
        p = 1 / (1 + np.exp(-z))
        return float(2 * p - 1)  # Convert back to [-1, 1]

    def calibrate_batch(self, predictions: List[float]) -> List[float]:
        """Calibrate a batch of predictions."""
        return [self.calibrate(p) for p in predictions]


class IsotonicCalibrator:
    """
    Isotonic regression calibration.

    Non-parametric calibration that learns a monotonic mapping.
    """

    def __init__(self):
        self.x_values: List[float] = []
        self.y_values: List[float] = []
        self._fitted = False

    def fit(self, predictions: List[float], outcomes: List[float]) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression.

        Uses Pool Adjacent Violators algorithm.
        """
        # Sort by prediction
        sorted_pairs = sorted(zip(predictions, outcomes))
        preds = [p[0] for p in sorted_pairs]
        outs = [p[1] for p in sorted_pairs]

        # Pool Adjacent Violators
        n = len(preds)
        y = list(outs)
        weights = [1.0] * n

        i = 0
        while i < n - 1:
            if y[i] > y[i + 1]:
                # Pool adjacent violators
                combined_y = (y[i] * weights[i] + y[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
                combined_weight = weights[i] + weights[i + 1]

                y[i] = combined_y
                weights[i] = combined_weight

                # Remove the pooled element
                y.pop(i + 1)
                weights.pop(i + 1)
                preds_list = list(preds)
                preds_list.pop(i + 1)
                preds = preds_list
                n -= 1

                # Go back and check again
                if i > 0:
                    i -= 1
            else:
                i += 1

        self.x_values = preds
        self.y_values = y
        self._fitted = True

        return self

    def calibrate(self, prediction: float) -> float:
        """
        Calibrate a single prediction using linear interpolation.
        """
        if not self._fitted or not self.x_values:
            return prediction

        # Handle edge cases
        if prediction <= self.x_values[0]:
            return self.y_values[0]
        if prediction >= self.x_values[-1]:
            return self.y_values[-1]

        # Find interval and interpolate
        for i in range(len(self.x_values) - 1):
            if self.x_values[i] <= prediction <= self.x_values[i + 1]:
                # Linear interpolation
                t = (prediction - self.x_values[i]) / (self.x_values[i + 1] - self.x_values[i])
                return self.y_values[i] + t * (self.y_values[i + 1] - self.y_values[i])

        return prediction

    def calibrate_batch(self, predictions: List[float]) -> List[float]:
        """Calibrate a batch of predictions."""
        return [self.calibrate(p) for p in predictions]


class GamePhaseCalibrator:
    """
    Separate calibration for different game phases.

    Early/mid/late game may have different calibration needs.
    """

    def __init__(self, num_phases: int = 3):
        self.num_phases = num_phases
        self.phase_calibrators: Dict[int, ValueCalibrator] = {
            i: ValueCalibrator() for i in range(num_phases)
        }
        self.phase_boundaries = [20, 50]  # Move thresholds

    def _get_phase(self, move_number: int) -> int:
        """Determine game phase from move number."""
        for i, boundary in enumerate(self.phase_boundaries):
            if move_number < boundary:
                return i
        return self.num_phases - 1

    def add_sample(self, prediction: float, outcome: float, move_number: int):
        """Add a sample with move number context."""
        phase = self._get_phase(move_number)
        self.phase_calibrators[phase].add_sample(prediction, outcome)

    def compute_calibration(self) -> Dict[int, CalibrationReport]:
        """Compute calibration for each phase."""
        return {
            phase: calibrator.compute_calibration()
            for phase, calibrator in self.phase_calibrators.items()
        }

    def find_optimal_temperatures(self) -> Dict[int, float]:
        """Find optimal temperature for each phase."""
        return {
            phase: calibrator.find_optimal_temperature()
            for phase, calibrator in self.phase_calibrators.items()
        }

    def calibrate_prediction(self, prediction: float, move_number: int) -> float:
        """Calibrate a prediction based on game phase."""
        phase = self._get_phase(move_number)
        return self.phase_calibrators[phase].calibrate_prediction(prediction)


class CalibrationTracker:
    """
    Track calibration metrics over training.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.history: List[CalibrationReport] = []
        self.running_predictions: List[float] = []
        self.running_outcomes: List[float] = []

    def add_sample(self, prediction: float, outcome: float):
        """Add a sample to the running window."""
        self.running_predictions.append(prediction)
        self.running_outcomes.append(outcome)

        # Trim to window size
        if len(self.running_predictions) > self.window_size:
            self.running_predictions.pop(0)
            self.running_outcomes.pop(0)

    def compute_current_calibration(self) -> Optional[CalibrationReport]:
        """Compute calibration for current window."""
        if len(self.running_predictions) < 100:
            return None

        calibrator = ValueCalibrator()
        calibrator.add_batch(self.running_predictions, self.running_outcomes)
        report = calibrator.compute_calibration()
        report.optimal_temperature = calibrator.find_optimal_temperature()

        self.history.append(report)
        return report

    def get_calibration_trend(self) -> Dict[str, List[float]]:
        """Get trend of calibration metrics over time."""
        return {
            'ece': [r.ece for r in self.history],
            'mce': [r.mce for r in self.history],
            'overconfidence': [r.overconfidence for r in self.history],
            'temperature': [r.optimal_temperature for r in self.history if r.optimal_temperature]
        }

    def is_well_calibrated(self, ece_threshold: float = 0.05) -> bool:
        """Check if model is well-calibrated."""
        if not self.history:
            return False
        return self.history[-1].ece < ece_threshold


def create_reliability_diagram(report: CalibrationReport, save_path: Optional[str] = None):
    """
    Create a reliability diagram visualization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for reliability diagram")
        return

    if not report.reliability_diagram:
        return

    rd = report.reliability_diagram
    midpoints = rd['bin_midpoints']
    outcomes = rd['mean_outcomes']
    counts = rd['counts']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.bar(midpoints, outcomes, width=0.08, alpha=0.7, label='Model')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Reliability Diagram (ECE={report.ece:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sample count histogram
    ax2.bar(midpoints, counts, width=0.08, alpha=0.7)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved reliability diagram to {save_path}")
    else:
        plt.show()


def main():
    """Demonstrate value calibration."""
    import random

    # Generate synthetic data
    # Simulate an overconfident model
    n_samples = 1000
    predictions = []
    outcomes = []

    for _ in range(n_samples):
        # True probability of winning
        true_prob = random.random()

        # Overconfident prediction (more extreme)
        noise = random.gauss(0, 0.1)
        pred = true_prob + (true_prob - 0.5) * 0.3 + noise
        pred = max(-1, min(1, 2 * pred - 1))  # Convert to [-1, 1] and clip

        # Outcome based on true probability
        outcome = 1 if random.random() < true_prob else -1

        predictions.append(pred)
        outcomes.append(outcome)

    # Analyze calibration
    calibrator = ValueCalibrator(num_bins=10)
    calibrator.add_batch(predictions, outcomes)

    print("=== Calibration Analysis ===")
    report = calibrator.compute_calibration()
    print(f"ECE (Expected Calibration Error): {report.ece:.4f}")
    print(f"MCE (Maximum Calibration Error): {report.mce:.4f}")
    print(f"Overconfidence: {report.overconfidence:.4f}")
    print(f"Brier Score: {report.brier_score:.4f}")

    # Find optimal temperature
    optimal_temp = calibrator.find_optimal_temperature()
    print(f"\nOptimal Temperature: {optimal_temp:.3f}")

    # Apply calibration and check improvement
    calibrated_predictions = calibrator.calibrate_batch(predictions)

    calibrator2 = ValueCalibrator(num_bins=10)
    calibrator2.add_batch(calibrated_predictions, outcomes)
    report2 = calibrator2.compute_calibration()

    print(f"\n=== After Temperature Scaling ===")
    print(f"ECE: {report2.ece:.4f}")
    print(f"MCE: {report2.mce:.4f}")
    print(f"Overconfidence: {report2.overconfidence:.4f}")

    # Test Platt scaling
    print("\n=== Platt Scaling ===")
    platt = PlattScaler()
    platt.fit(predictions, outcomes)
    print(f"Fitted parameters: A={platt.a:.4f}, B={platt.b:.4f}")

    platt_calibrated = platt.calibrate_batch(predictions)
    calibrator3 = ValueCalibrator(num_bins=10)
    calibrator3.add_batch(platt_calibrated, outcomes)
    report3 = calibrator3.compute_calibration()
    print(f"ECE after Platt scaling: {report3.ece:.4f}")

    # Test isotonic calibration
    print("\n=== Isotonic Calibration ===")
    isotonic = IsotonicCalibrator()
    isotonic.fit(predictions, outcomes)

    isotonic_calibrated = isotonic.calibrate_batch(predictions)
    calibrator4 = ValueCalibrator(num_bins=10)
    calibrator4.add_batch(isotonic_calibrated, outcomes)
    report4 = calibrator4.compute_calibration()
    print(f"ECE after Isotonic calibration: {report4.ece:.4f}")

    # Create reliability diagram
    print("\nGenerating reliability diagram...")
    try:
        create_reliability_diagram(report, 'reliability_diagram.png')
    except Exception as e:
        print(f"Could not create diagram: {e}")

    # Test phase-specific calibration
    print("\n=== Phase-Specific Calibration ===")
    phase_calibrator = GamePhaseCalibrator()

    for i, (pred, out) in enumerate(zip(predictions, outcomes)):
        move_number = random.randint(0, 100)
        phase_calibrator.add_sample(pred, out, move_number)

    phase_reports = phase_calibrator.compute_calibration()
    for phase, report in phase_reports.items():
        print(f"Phase {phase}: ECE={report.ece:.4f}, samples={report.total_samples}")

    phase_temps = phase_calibrator.find_optimal_temperatures()
    print(f"Phase temperatures: {phase_temps}")


if __name__ == "__main__":
    main()
