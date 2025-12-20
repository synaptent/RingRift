"""Uncertainty Calibration Study for GMO AI.

Evaluates whether the GMO model's uncertainty estimates are well-calibrated:
- High predicted uncertainty should correlate with high prediction errors
- Confidence intervals should have correct coverage

Metrics computed:
1. Error-Uncertainty Correlation: Spearman correlation between |error| and variance
2. Calibration Curve: Binned expected vs actual error
3. Coverage: % of outcomes within 1/2 sigma confidence intervals
4. Brier Score: For win probability calibration
"""

from __future__ import annotations

import argparse
import json
import logging

# Add parent to path for imports
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai.factory import AIFactory
from app.ai.gmo_ai import GMOAI, GMOConfig
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType
from app.training.train_gmo_selfplay import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """Single calibration sample."""
    predicted_value: float  # Model's value prediction
    predicted_variance: float  # Model's uncertainty (variance)
    actual_outcome: float  # True game outcome (-1, 0, or 1)
    move_idx: int  # Move number in game
    game_id: int  # Game identifier


@dataclass
class CalibrationMetrics:
    """Computed calibration metrics."""
    num_samples: int
    error_uncertainty_correlation: float  # Spearman correlation
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_error: float
    mean_predicted_variance: float
    coverage_1_sigma: float  # % within 1 std
    coverage_2_sigma: float  # % within 2 std
    calibration_error: float  # ECE-style metric
    brier_score: float  # For win probability
    uncertainty_bins: list[tuple[float, float, float]]  # (bin_center, mean_variance, mean_error)


def collect_samples_from_game(
    gmo_ai: GMOAI,
    opponent_ai,
    game_id: int,
    max_moves: int = 500
) -> list[CalibrationSample]:
    """Play a game and collect calibration samples.

    Returns samples of (predicted_value, predicted_variance, actual_outcome).
    """
    samples = []
    state = create_initial_state(
        game_id=f"calibration_{game_id}",
        board_type=BoardType.SQUARE8,
        rng_seed=game_id,
    )
    gmo_player_num = 1  # GMO plays as Player 1
    move_idx = 0

    while state.game_status.value == "active" and move_idx < max_moves:
        current_player = state.current_player
        legal_moves = GameEngine.get_valid_moves(state, current_player)

        if not legal_moves:
            # Check for bookkeeping move requirements (NO_LINE_ACTION, etc.)
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_idx += 1
                    continue
            # No valid moves and no bookkeeping required - game stuck
            break

        if current_player == gmo_player_num:
            # Get GMO's predictions with uncertainty
            predictions = gmo_ai.get_move_predictions_with_uncertainty(
                state, legal_moves
            )

            # Record prediction for chosen move
            best_idx = max(range(len(predictions)), key=lambda i: predictions[i][0])
            best_value, best_variance = predictions[best_idx]

            # Store sample (outcome filled in later)
            samples.append(CalibrationSample(
                predicted_value=best_value,
                predicted_variance=best_variance,
                actual_outcome=0.0,  # Will be filled after game ends
                move_idx=move_idx,
                game_id=game_id
            ))

            # Make the move
            move = gmo_ai.select_move(state)
        else:
            # Opponent's turn
            move = opponent_ai.select_move(state)

        if move is None:
            # Check for bookkeeping move requirements
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_idx += 1

    # Determine game outcome
    if state.game_status.value != "active":
        if state.winner == gmo_player_num:
            outcome = 1.0
        elif state.winner is None:
            outcome = 0.0  # Draw
        else:
            outcome = -1.0
    else:
        outcome = 0.0  # Game didn't finish

    # Fill in outcomes for all samples
    for sample in samples:
        sample.actual_outcome = outcome

    return samples


def compute_calibration_metrics(samples: list[CalibrationSample]) -> CalibrationMetrics:
    """Compute calibration metrics from samples."""
    if not samples:
        raise ValueError("No samples provided")

    predictions = np.array([s.predicted_value for s in samples])
    variances = np.array([s.predicted_variance for s in samples])
    outcomes = np.array([s.actual_outcome for s in samples])

    # Basic error metrics
    errors = predictions - outcomes
    abs_errors = np.abs(errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    mean_var = np.mean(variances)

    # Spearman correlation between |error| and variance
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(abs_errors, variances)
    if np.isnan(correlation):
        correlation = 0.0

    # Coverage metrics (what % of outcomes fall within confidence intervals)
    std_devs = np.sqrt(np.maximum(variances, 1e-6))
    z_scores = np.abs(errors) / std_devs
    coverage_1_sigma = np.mean(z_scores <= 1.0)  # Should be ~68%
    coverage_2_sigma = np.mean(z_scores <= 2.0)  # Should be ~95%

    # Calibration error (binned)
    # Sort by variance and bin
    n_bins = 10
    sorted_indices = np.argsort(variances)
    bin_size = len(samples) // n_bins

    calibration_errors = []
    uncertainty_bins = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(samples)
        bin_indices = sorted_indices[start:end]

        bin_variance = np.mean(variances[bin_indices])
        bin_error = np.mean(abs_errors[bin_indices])
        bin_expected_error = np.sqrt(bin_variance)  # Expected |error| ~ sqrt(variance)

        calibration_errors.append(np.abs(bin_error - bin_expected_error))
        uncertainty_bins.append((
            float(np.mean(np.sqrt(variances[bin_indices]))),  # bin center (std dev)
            float(bin_variance),
            float(bin_error)
        ))

    calibration_error = np.mean(calibration_errors)

    # Brier score for win probability
    # Convert predictions to win probabilities: p_win = (pred + 1) / 2
    win_probs = (predictions + 1) / 2
    actual_wins = (outcomes + 1) / 2
    brier_score = np.mean((win_probs - actual_wins) ** 2)

    return CalibrationMetrics(
        num_samples=len(samples),
        error_uncertainty_correlation=float(correlation),
        mean_squared_error=float(mse),
        root_mean_squared_error=float(rmse),
        mean_absolute_error=float(mae),
        mean_predicted_variance=float(mean_var),
        coverage_1_sigma=float(coverage_1_sigma),
        coverage_2_sigma=float(coverage_2_sigma),
        calibration_error=float(calibration_error),
        brier_score=float(brier_score),
        uncertainty_bins=uncertainty_bins
    )


def run_calibration_study(
    checkpoint_path: str | None = None,
    num_games: int = 50,
    opponent_type: str = "random",
    output_dir: str = "data/calibration"
) -> CalibrationMetrics:
    """Run uncertainty calibration study.

    Args:
        checkpoint_path: Path to GMO checkpoint
        num_games: Number of games to play for calibration
        opponent_type: Type of opponent ("random" or "heuristic")
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load GMO AI (player 1)
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOConfig()
    gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Try default path
        default_path = "models/gmo/gmo_best.pt"
        if Path(default_path).exists():
            gmo_ai.load_checkpoint(default_path)
            logger.info(f"Loaded checkpoint from {default_path}")

    # Create opponent (player 2)
    opponent_config = AIConfig(difficulty=2)
    if opponent_type == "random":
        opponent = AIFactory.create(AIType.RANDOM, player_number=2, config=opponent_config)
    elif opponent_type == "heuristic":
        opponent = AIFactory.create(AIType.HEURISTIC, player_number=2, config=opponent_config)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    logger.info(f"Running calibration study: {num_games} games vs {opponent_type}")

    # Collect samples from games
    all_samples = []
    wins, losses, draws = 0, 0, 0

    for game_id in range(num_games):
        samples = collect_samples_from_game(gmo_ai, opponent, game_id)
        all_samples.extend(samples)

        if samples:
            outcome = samples[0].actual_outcome
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1

        if (game_id + 1) % 10 == 0:
            logger.info(f"Game {game_id + 1}/{num_games}: {len(all_samples)} samples collected")

    logger.info(f"Game results: {wins}W/{losses}L/{draws}D ({100*wins/num_games:.1f}% win rate)")
    logger.info(f"Total samples collected: {len(all_samples)}")

    # Compute metrics
    if not all_samples:
        logger.error("No samples collected!")
        return None

    metrics = compute_calibration_metrics(all_samples)

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Samples: {metrics.num_samples}")
    logger.info(f"Error-Uncertainty Correlation: {metrics.error_uncertainty_correlation:.4f}")
    logger.info("  (Ideal: positive; high uncertainty should predict high error)")
    logger.info(f"MSE: {metrics.mean_squared_error:.4f}")
    logger.info(f"RMSE: {metrics.root_mean_squared_error:.4f}")
    logger.info(f"MAE: {metrics.mean_absolute_error:.4f}")
    logger.info(f"Mean Predicted Variance: {metrics.mean_predicted_variance:.4f}")
    logger.info(f"Coverage (1σ): {100*metrics.coverage_1_sigma:.1f}% (ideal: 68%)")
    logger.info(f"Coverage (2σ): {100*metrics.coverage_2_sigma:.1f}% (ideal: 95%)")
    logger.info(f"Calibration Error: {metrics.calibration_error:.4f}")
    logger.info(f"Brier Score: {metrics.brier_score:.4f} (lower is better)")

    logger.info("\nUncertainty Bins (std_dev, variance, actual_error):")
    for i, (std_dev, var, err) in enumerate(metrics.uncertainty_bins):
        expected_err = np.sqrt(var)
        logger.info(f"  Bin {i+1}: σ={std_dev:.3f}, var={var:.4f}, err={err:.4f} (expected: {expected_err:.4f})")

    # Interpret results
    logger.info("\n" + "-" * 60)
    logger.info("INTERPRETATION")
    logger.info("-" * 60)

    if metrics.error_uncertainty_correlation > 0.3:
        logger.info("✓ Good: Uncertainty correlates with error (well-calibrated)")
    elif metrics.error_uncertainty_correlation > 0.1:
        logger.info("○ Moderate: Some correlation between uncertainty and error")
    else:
        logger.info("✗ Poor: Uncertainty doesn't predict error well (overconfident)")

    if 0.60 <= metrics.coverage_1_sigma <= 0.76:
        logger.info("✓ Good: 1σ coverage is near ideal 68%")
    elif metrics.coverage_1_sigma < 0.60:
        logger.info("✗ Overconfident: 1σ coverage too low (underestimates uncertainty)")
    else:
        logger.info("○ Underconfident: 1σ coverage too high (overestimates uncertainty)")

    if 0.90 <= metrics.coverage_2_sigma <= 0.98:
        logger.info("✓ Good: 2σ coverage is near ideal 95%")
    elif metrics.coverage_2_sigma < 0.90:
        logger.info("✗ Overconfident: 2σ coverage too low")
    else:
        logger.info("○ Underconfident: 2σ coverage too high")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "num_games": num_games,
        "opponent_type": opponent_type,
        "game_results": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0
        },
        "metrics": {
            "num_samples": metrics.num_samples,
            "error_uncertainty_correlation": metrics.error_uncertainty_correlation,
            "mse": metrics.mean_squared_error,
            "rmse": metrics.root_mean_squared_error,
            "mae": metrics.mean_absolute_error,
            "mean_predicted_variance": metrics.mean_predicted_variance,
            "coverage_1_sigma": metrics.coverage_1_sigma,
            "coverage_2_sigma": metrics.coverage_2_sigma,
            "calibration_error": metrics.calibration_error,
            "brier_score": metrics.brier_score
        },
        "uncertainty_bins": [
            {"std_dev": std_dev, "variance": var, "actual_error": err}
            for std_dev, var, err in metrics.uncertainty_bins
        ]
    }

    results_file = output_path / f"calibration_{opponent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return metrics


def find_optimal_temperature(
    samples: list[CalibrationSample],
    temperatures: list[float] | None = None,
) -> tuple[float, float]:
    """Find optimal temperature scaling for uncertainty calibration.

    Uses grid search to find temperature that gives coverage closest to 68%
    for 1-sigma confidence intervals.

    Args:
        samples: Calibration samples with predictions and outcomes
        temperatures: List of temperatures to try (default: [0.5, 1, 2, 5, 10, 20, 50])

    Returns:
        (optimal_temperature, best_coverage_error)
    """
    if temperatures is None:
        # Common range for overconfident models (T > 1)
        temperatures = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    predictions = np.array([s.predicted_value for s in samples])
    variances = np.array([s.predicted_variance for s in samples])
    outcomes = np.array([s.actual_outcome for s in samples])

    errors = np.abs(predictions - outcomes)

    best_temp = 1.0
    best_error = float('inf')

    for temp in temperatures:
        # Apply temperature scaling
        calibrated_variance = variances * temp
        calibrated_std = np.sqrt(np.maximum(calibrated_variance, 1e-6))

        # Compute coverage
        z_scores = errors / calibrated_std
        coverage_1_sigma = np.mean(z_scores <= 1.0)

        # Error from ideal 68% coverage
        coverage_error = abs(coverage_1_sigma - 0.68)

        if coverage_error < best_error:
            best_error = coverage_error
            best_temp = temp

    return best_temp, best_error


def run_temperature_calibration(
    checkpoint_path: str | None = None,
    num_games: int = 50,
    opponent_type: str = "random",
    output_dir: str = "data/calibration"
) -> float:
    """Run temperature calibration to find optimal uncertainty scaling.

    Returns the optimal temperature that should be set in GMOConfig.calibration_temperature.

    Args:
        checkpoint_path: Path to GMO checkpoint
        num_games: Number of games to collect calibration samples
        opponent_type: Type of opponent
        output_dir: Output directory

    Returns:
        Optimal calibration temperature
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load GMO AI
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOConfig()
    gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        default_path = "models/gmo/gmo_best.pt"
        if Path(default_path).exists():
            gmo_ai.load_checkpoint(default_path)
            logger.info(f"Loaded checkpoint from {default_path}")

    # Create opponent
    opponent_config = AIConfig(difficulty=2)
    if opponent_type == "random":
        opponent = AIFactory.create(AIType.RANDOM, player_number=2, config=opponent_config)
    else:
        opponent = AIFactory.create(AIType.HEURISTIC, player_number=2, config=opponent_config)

    logger.info(f"Collecting calibration samples: {num_games} games vs {opponent_type}")

    # Collect samples
    all_samples = []
    for game_id in range(num_games):
        samples = collect_samples_from_game(gmo_ai, opponent, game_id)
        all_samples.extend(samples)

        if (game_id + 1) % 10 == 0:
            logger.info(f"Game {game_id + 1}/{num_games}: {len(all_samples)} samples")

    if not all_samples:
        logger.error("No samples collected!")
        return 1.0

    logger.info(f"Total samples: {len(all_samples)}")

    # Find optimal temperature
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    optimal_temp, coverage_error = find_optimal_temperature(all_samples, temperatures)

    logger.info("\n" + "=" * 60)
    logger.info("TEMPERATURE CALIBRATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Optimal temperature: {optimal_temp}")
    logger.info(f"Coverage error from 68% target: {coverage_error:.4f}")

    # Show coverage at different temperatures
    predictions = np.array([s.predicted_value for s in all_samples])
    variances = np.array([s.predicted_variance for s in all_samples])
    outcomes = np.array([s.actual_outcome for s in all_samples])
    errors = np.abs(predictions - outcomes)

    logger.info("\nTemperature sweep:")
    for temp in temperatures:
        calibrated_variance = variances * temp
        calibrated_std = np.sqrt(np.maximum(calibrated_variance, 1e-6))
        z_scores = errors / calibrated_std
        coverage = np.mean(z_scores <= 1.0)
        marker = " <-- OPTIMAL" if temp == optimal_temp else ""
        logger.info(f"  T={temp:6.1f}: 1σ coverage = {100*coverage:.1f}%{marker}")

    logger.info("\nTo use this calibration, set in your code:")
    logger.info(f"  gmo_config = GMOConfig(calibration_temperature={optimal_temp})")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "num_games": num_games,
        "num_samples": len(all_samples),
        "optimal_temperature": optimal_temp,
        "coverage_error": coverage_error,
    }

    results_file = output_path / f"temperature_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return optimal_temp


def main():
    parser = argparse.ArgumentParser(description="GMO Uncertainty Calibration Study")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to GMO checkpoint")
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games to play")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "heuristic"],
                        help="Opponent type")
    parser.add_argument("--output-dir", type=str, default="data/calibration",
                        help="Output directory for results")
    parser.add_argument("--calibrate", action="store_true",
                        help="Find optimal temperature for uncertainty calibration")

    args = parser.parse_args()

    if args.calibrate:
        optimal_temp = run_temperature_calibration(
            checkpoint_path=args.checkpoint,
            num_games=args.games,
            opponent_type=args.opponent,
            output_dir=args.output_dir
        )
        print(f"\nOptimal temperature: {optimal_temp}")
    else:
        run_calibration_study(
            checkpoint_path=args.checkpoint,
            num_games=args.games,
            opponent_type=args.opponent,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
