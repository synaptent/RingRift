#!/usr/bin/env python3
"""GMO Uncertainty Calibration - Validate uncertainty estimates.

This script evaluates whether GMO's uncertainty estimates are well-calibrated:
- High uncertainty should correlate with unpredictable outcomes
- Confident predictions should be more accurate

Metrics computed:
1. Correlation between predicted uncertainty and actual outcome variance
2. Calibration plot: predicted confidence vs actual accuracy
3. Expected Calibration Error (ECE)
4. Brier score decomposition

Usage:
    python scripts/gmo_uncertainty_calibration.py --games 100
    python scripts/gmo_uncertainty_calibration.py --plot
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from app.ai.gmo_ai import GMOAI, GMOConfig
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameStatus
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GMO_CHECKPOINT = PROJECT_ROOT / "models" / "gmo" / "gmo_best.pt"
RESULTS_DIR = PROJECT_ROOT / "data" / "gmo_calibration"


@dataclass
class UncertaintyRecord:
    """Record of a single prediction with uncertainty."""
    predicted_value: float
    predicted_variance: float
    actual_outcome: float  # 1.0 = win, -1.0 = loss, 0.0 = draw
    move_number: int
    game_id: int


def create_gmo(player_number: int, device: str = "cpu") -> GMOAI:
    """Create GMO AI with trained checkpoint."""
    ai_config = AIConfig(difficulty=5)
    gmo_config = GMOConfig(device=device)
    ai = GMOAI(player_number=player_number, config=ai_config, gmo_config=gmo_config)
    if GMO_CHECKPOINT.exists():
        ai.load_checkpoint(GMO_CHECKPOINT)
    return ai


def collect_predictions(
    num_games: int = 100,
    device: str = "cpu",
    num_players: int = 2,
) -> list[UncertaintyRecord]:
    """Collect GMO predictions and actual outcomes.

    Plays games and records:
    - Predicted value for each move
    - Predicted variance (uncertainty)
    - Actual game outcome
    """
    from app.ai.random_ai import RandomAI

    records = []

    for game_id in range(num_games):
        # Create AIs with unique seeds per game
        gmo = create_gmo(1, device)
        opponent = RandomAI(player_number=2, config=AIConfig(difficulty=1, rng_seed=game_id))

        state = create_initial_state(board_type=BoardType.SQUARE8, num_players=num_players)
        game_predictions = []

        move_num = 0
        while state.game_status == GameStatus.ACTIVE:
            current_player = state.current_player
            legal_moves = GameEngine.get_valid_moves(state, current_player)
            if not legal_moves:
                # Check for phase requirements (no-action moves)
                phase_req = GameEngine.get_phase_requirement(state, current_player)
                if phase_req:
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(phase_req, state)
                    state = GameEngine.apply_move(state, bookkeeping_move)
                    move_num += 1
                    continue
                break

            if current_player == 1:
                # GMO's turn - record prediction
                with torch.no_grad():
                    state_embed = gmo.state_encoder(state).to(gmo.device)

                # Get prediction for best move candidate
                best_score = float('-inf')
                best_value = 0.0
                best_var = 0.0

                for move in legal_moves[:min(5, len(legal_moves))]:  # Check top candidates
                    with torch.no_grad():
                        move_embed = gmo.move_encoder(move).to(gmo.device)

                    # Get value and uncertainty
                    mean_val, _, var = gmo._estimate_uncertainty(state_embed, move_embed)
                    score = mean_val.item()

                    if score > best_score:
                        best_score = score
                        best_value = mean_val.item()
                        best_var = var.item()

                game_predictions.append({
                    "value": best_value,
                    "variance": best_var,
                    "move_num": move_num,
                })

                move = gmo.select_move(state)
            else:
                move = opponent.select_move(state)

            if move is None:
                phase_req = GameEngine.get_phase_requirement(state, current_player)
                if phase_req is None:
                    break
                move = GameEngine.synthesize_bookkeeping_move(phase_req, state)

            state = GameEngine.apply_move(state, move)
            move_num += 1

        # Determine outcome
        if state.game_status == GameStatus.COMPLETED:
            if state.winner == 1:
                outcome = 1.0
            elif state.winner == 2:
                outcome = -1.0
            else:
                outcome = 0.0
        else:
            outcome = 0.0

        # Create records
        for pred in game_predictions:
            records.append(UncertaintyRecord(
                predicted_value=pred["value"],
                predicted_variance=pred["variance"],
                actual_outcome=outcome,
                move_number=pred["move_num"],
                game_id=game_id,
            ))

        if (game_id + 1) % 20 == 0:
            logger.info(f"Collected {game_id + 1}/{num_games} games")

    return records


def compute_calibration_metrics(records: list[UncertaintyRecord]) -> dict:
    """Compute calibration metrics from prediction records."""
    if not records:
        return {}

    # Extract arrays
    values = np.array([r.predicted_value for r in records])
    variances = np.array([r.predicted_variance for r in records])
    outcomes = np.array([r.actual_outcome for r in records])

    # 1. Correlation: variance vs squared error
    squared_errors = (values - outcomes) ** 2
    corr_var_error = np.corrcoef(variances, squared_errors)[0, 1]

    # 2. Correlation: confidence vs accuracy
    # High confidence = low variance
    confidences = 1.0 / (1.0 + variances)
    # Accuracy = 1 if prediction sign matches outcome sign
    accuracies = ((values > 0) == (outcomes > 0)).astype(float)
    corr_conf_acc = np.corrcoef(confidences, accuracies)[0, 1]

    # 3. Expected Calibration Error (ECE)
    # Bin predictions by confidence, compare avg confidence to accuracy
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(num_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / len(records)) * abs(bin_acc - bin_conf)
            bin_stats.append({
                "bin": i,
                "confidence": float(bin_conf),
                "accuracy": float(bin_acc),
                "count": int(bin_size),
            })

    # 4. Brier score
    # For binary outcomes (win vs not-win)
    win_probs = (values + 1) / 2  # Map [-1, 1] to [0, 1]
    win_actual = (outcomes > 0).astype(float)
    brier_score = ((win_probs - win_actual) ** 2).mean()

    # 5. Uncertainty calibration by move phase
    early_game_mask = np.array([r.move_number < 10 for r in records])
    mid_game_mask = np.array([(10 <= r.move_number < 30) for r in records])
    late_game_mask = np.array([r.move_number >= 30 for r in records])

    phase_stats = {}
    for phase_name, mask in [("early", early_game_mask), ("mid", mid_game_mask), ("late", late_game_mask)]:
        if mask.sum() > 0:
            phase_var = variances[mask].mean()
            phase_err = squared_errors[mask].mean()
            phase_stats[phase_name] = {
                "avg_variance": float(phase_var),
                "avg_squared_error": float(phase_err),
                "count": int(mask.sum()),
            }

    return {
        "correlation_variance_error": float(corr_var_error),
        "correlation_confidence_accuracy": float(corr_conf_acc),
        "expected_calibration_error": float(ece),
        "brier_score": float(brier_score),
        "num_records": len(records),
        "avg_variance": float(variances.mean()),
        "avg_squared_error": float(squared_errors.mean()),
        "bin_statistics": bin_stats,
        "phase_statistics": phase_stats,
    }


def print_calibration_report(metrics: dict) -> None:
    """Print formatted calibration report."""
    print("\n" + "="*60)
    print("UNCERTAINTY CALIBRATION REPORT")
    print("="*60)

    print(f"\nSample size: {metrics['num_records']} predictions")

    print("\n--- Correlation Metrics ---")
    corr_ve = metrics["correlation_variance_error"]
    corr_ca = metrics["correlation_confidence_accuracy"]
    print(f"Variance vs Squared Error: {corr_ve:.3f}")
    print(f"  (Positive = high uncertainty predicts larger errors) {'OK' if corr_ve > 0.1 else 'WEAK'}")
    print(f"Confidence vs Accuracy: {corr_ca:.3f}")
    print(f"  (Positive = confident predictions more accurate) {'OK' if corr_ca > 0.1 else 'WEAK'}")

    print("\n--- Calibration Metrics ---")
    ece = metrics["expected_calibration_error"]
    brier = metrics["brier_score"]
    print(f"Expected Calibration Error (ECE): {ece:.3f}")
    print(f"  (Lower is better, <0.1 is well-calibrated) {'GOOD' if ece < 0.1 else 'MODERATE' if ece < 0.2 else 'POOR'}")
    print(f"Brier Score: {brier:.3f}")
    print(f"  (Lower is better, 0.25 is random) {'GOOD' if brier < 0.2 else 'MODERATE' if brier < 0.3 else 'POOR'}")

    print("\n--- Calibration by Game Phase ---")
    for phase, stats in metrics.get("phase_statistics", {}).items():
        var_err_ratio = stats["avg_variance"] / max(stats["avg_squared_error"], 0.001)
        print(f"{phase.upper()} game (n={stats['count']}):")
        print(f"  Avg variance: {stats['avg_variance']:.4f}")
        print(f"  Avg squared error: {stats['avg_squared_error']:.4f}")
        print(f"  Ratio (var/err): {var_err_ratio:.2f} (ideal ~1.0)")

    print("\n--- Confidence Bins ---")
    print(f"{'Bin':<6} {'Confidence':<12} {'Accuracy':<12} {'Count':<8}")
    print("-"*40)
    for bin_stat in metrics.get("bin_statistics", []):
        print(f"{bin_stat['bin']:<6} {bin_stat['confidence']:<12.3f} {bin_stat['accuracy']:<12.3f} {bin_stat['count']:<8}")


def create_calibration_plot(metrics: dict, output_path: Path) -> None:
    """Create calibration plot (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    bin_stats = metrics.get("bin_statistics", [])
    if not bin_stats:
        return

    confidences = [b["confidence"] for b in bin_stats]
    accuracies = [b["accuracy"] for b in bin_stats]

    _fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Actual calibration
    ax.scatter(confidences, accuracies, s=100, c='blue', alpha=0.7)
    ax.plot(confidences, accuracies, 'b-', alpha=0.5, label='GMO calibration')

    ax.set_xlabel('Predicted Confidence', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title('GMO Uncertainty Calibration', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved calibration plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GMO Uncertainty Calibration")
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to analyze"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "calibration_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Create calibration plot"
    )

    args = parser.parse_args()

    logger.info(f"Collecting predictions from {args.games} games...")
    start_time = time.time()

    records = collect_predictions(args.games, args.device)
    logger.info(f"Collected {len(records)} predictions in {time.time() - start_time:.1f}s")

    metrics = compute_calibration_metrics(records)
    metrics["games_analyzed"] = args.games
    metrics["collection_time_sec"] = time.time() - start_time

    print_calibration_report(metrics)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Create plot if requested
    if args.plot:
        plot_path = args.output.with_suffix(".png")
        create_calibration_plot(metrics, plot_path)

    # Summary verdict
    print("\n" + "="*60)
    print("CALIBRATION VERDICT")
    print("="*60)

    issues = []
    if metrics["correlation_variance_error"] < 0.1:
        issues.append("Variance doesn't predict errors well")
    if metrics["expected_calibration_error"] > 0.2:
        issues.append("Confidence bins poorly calibrated")
    if metrics["brier_score"] > 0.3:
        issues.append("Probabilistic predictions inaccurate")

    if not issues:
        print("GMO uncertainty estimates are WELL-CALIBRATED")
    else:
        print("GMO uncertainty has calibration issues:")
        for issue in issues:
            print(f"  - {issue}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
