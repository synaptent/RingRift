#!/usr/bin/env python3
"""Multi-Objective Optimization for RingRift AI Training.

Balances Elo improvement against training cost (GPU hours) using Pareto
optimization. Finds the optimal trade-off between playing strength and
resource efficiency.

Key Features:
- Pareto frontier tracking (Elo vs GPU-hours)
- Resource allocation across configs
- Cost-aware hyperparameter selection
- Efficiency recommendations

Usage:
    # Analyze current training efficiency
    python scripts/multi_objective_optimizer.py --analyze

    # Get allocation recommendations
    python scripts/multi_objective_optimizer.py --recommend --budget 100

    # Run optimization sweep
    python scripts/multi_objective_optimizer.py --optimize \
        --configs square8_2p square8_3p \
        --budget 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("multi_objective_optimizer", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics for a training run."""
    config: str
    elo_start: float
    elo_end: float
    gpu_hours: float
    games_generated: int
    training_samples: int
    timestamp: str

    @property
    def elo_gain(self) -> float:
        return self.elo_end - self.elo_start

    @property
    def elo_per_gpu_hour(self) -> float:
        if self.gpu_hours <= 0:
            return 0.0
        return self.elo_gain / self.gpu_hours

    @property
    def games_per_gpu_hour(self) -> float:
        if self.gpu_hours <= 0:
            return 0.0
        return self.games_generated / self.gpu_hours


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    config: str
    elo: float
    gpu_hours: float
    efficiency: float  # Elo per GPU hour
    is_pareto_optimal: bool = True
    dominated_by: List[str] = field(default_factory=list)


@dataclass
class AllocationRecommendation:
    """Resource allocation recommendation."""
    config: str
    recommended_gpu_hours: float
    expected_elo_gain: float
    priority: str  # "high", "medium", "low"
    rationale: str


@dataclass
class OptimizationResult:
    """Results of multi-objective optimization."""
    pareto_frontier: List[ParetoPoint]
    recommendations: List[AllocationRecommendation]
    total_budget: float
    allocated_budget: float
    expected_total_elo_gain: float
    timestamp: str


# Cost estimates (adjust based on actual infrastructure)
GPU_COST_PER_HOUR_USD = {
    "A100": 3.00,
    "V100": 1.50,
    "T4": 0.50,
    "M1_Mac": 0.10,  # Local development
    "default": 1.00,
}


def load_training_history(db_path: Path) -> List[TrainingMetrics]:
    """Load training history from database."""
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    metrics = []

    try:
        cursor = conn.execute("""
            SELECT
                board_type || '_' || num_players || 'p' as config,
                initial_elo as elo_start,
                final_elo as elo_end,
                gpu_hours,
                games_generated,
                training_samples,
                completed_at as timestamp
            FROM training_history
            WHERE completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT 100
        """)

        for row in cursor:
            metrics.append(TrainingMetrics(
                config=row["config"],
                elo_start=row["elo_start"] or 1500.0,
                elo_end=row["elo_end"] or 1500.0,
                gpu_hours=row["gpu_hours"] or 0.0,
                games_generated=row["games_generated"] or 0,
                training_samples=row["training_samples"] or 0,
                timestamp=row["timestamp"] or "",
            ))
    except sqlite3.OperationalError as e:
        logger.warning(f"Error reading training history: {e}")

    conn.close()
    return metrics


def compute_pareto_frontier(metrics: List[TrainingMetrics]) -> List[ParetoPoint]:
    """Compute Pareto frontier from training metrics.

    A point is Pareto optimal if no other point is better in all objectives
    (higher Elo and lower GPU cost).
    """
    # Group by config and get best results
    config_best: Dict[str, Tuple[float, float]] = {}  # config -> (max_elo, min_gpu_hours)

    for m in metrics:
        if m.config not in config_best:
            config_best[m.config] = (m.elo_end, m.gpu_hours)
        else:
            prev_elo, prev_gpu = config_best[m.config]
            # Update if this run achieved higher Elo
            if m.elo_end > prev_elo:
                config_best[m.config] = (m.elo_end, m.gpu_hours)

    # Create points and compute efficiency
    points = []
    for config, (elo, gpu_hours) in config_best.items():
        efficiency = elo / max(gpu_hours, 0.1) if gpu_hours > 0 else 0.0
        points.append(ParetoPoint(
            config=config,
            elo=elo,
            gpu_hours=gpu_hours,
            efficiency=efficiency,
        ))

    # Find Pareto optimal points
    for i, p1 in enumerate(points):
        p1.is_pareto_optimal = True
        p1.dominated_by = []
        for j, p2 in enumerate(points):
            if i == j:
                continue
            # p2 dominates p1 if p2 has higher elo AND lower/equal gpu hours
            # OR equal elo and strictly lower gpu hours
            if (p2.elo > p1.elo and p2.gpu_hours <= p1.gpu_hours) or \
               (p2.elo >= p1.elo and p2.gpu_hours < p1.gpu_hours):
                p1.is_pareto_optimal = False
                p1.dominated_by.append(p2.config)

    return sorted(points, key=lambda p: -p.elo)


def estimate_marginal_elo_gain(
    config: str,
    current_elo: float,
    historical_efficiency: float,
    additional_gpu_hours: float,
) -> float:
    """Estimate Elo gain from additional training.

    Uses diminishing returns model: gain = k * sqrt(hours)
    where k is estimated from historical efficiency.
    """
    # Diminishing returns - Elo gain is proportional to sqrt of investment
    # This models the empirical observation that doubling training
    # doesn't double Elo gain
    k = historical_efficiency * 0.5  # Adjust coefficient based on observations

    # Higher current Elo = harder to improve
    elo_factor = max(0.5, 1.0 - (current_elo - 1500) / 1000)

    estimated_gain = k * np.sqrt(additional_gpu_hours) * elo_factor

    return max(0, estimated_gain)


def allocate_budget(
    pareto_points: List[ParetoPoint],
    total_budget: float,
    strategy: str = "balanced",
) -> List[AllocationRecommendation]:
    """Allocate GPU budget across configs.

    Strategies:
    - "balanced": Equal allocation to all configs
    - "efficiency": Prioritize high-efficiency configs
    - "underperforming": Focus on configs with most improvement potential
    - "pareto": Only invest in Pareto-optimal configs
    """
    recommendations = []

    if not pareto_points:
        return recommendations

    if strategy == "balanced":
        budget_per_config = total_budget / len(pareto_points)
        for point in pareto_points:
            expected_gain = estimate_marginal_elo_gain(
                point.config, point.elo, point.efficiency, budget_per_config
            )
            recommendations.append(AllocationRecommendation(
                config=point.config,
                recommended_gpu_hours=budget_per_config,
                expected_elo_gain=expected_gain,
                priority="medium",
                rationale=f"Balanced allocation strategy",
            ))

    elif strategy == "efficiency":
        # Weight by efficiency
        total_efficiency = sum(p.efficiency for p in pareto_points)
        if total_efficiency > 0:
            for point in pareto_points:
                weight = point.efficiency / total_efficiency
                allocation = total_budget * weight
                expected_gain = estimate_marginal_elo_gain(
                    point.config, point.elo, point.efficiency, allocation
                )
                priority = "high" if weight > 0.3 else "medium" if weight > 0.1 else "low"
                recommendations.append(AllocationRecommendation(
                    config=point.config,
                    recommended_gpu_hours=allocation,
                    expected_elo_gain=expected_gain,
                    priority=priority,
                    rationale=f"Efficiency-weighted: {point.efficiency:.2f} Elo/GPU-hr",
                ))

    elif strategy == "underperforming":
        # Identify configs that are underperforming relative to others
        avg_elo = np.mean([p.elo for p in pareto_points])
        for point in pareto_points:
            # More budget to configs below average
            if point.elo < avg_elo:
                weight = 1.5
                priority = "high"
                rationale = f"Below average Elo ({point.elo:.0f} < {avg_elo:.0f})"
            else:
                weight = 0.5
                priority = "low"
                rationale = f"Above average Elo ({point.elo:.0f} > {avg_elo:.0f})"

            total_weight = sum(1.5 if p.elo < avg_elo else 0.5 for p in pareto_points)
            allocation = total_budget * weight / total_weight
            expected_gain = estimate_marginal_elo_gain(
                point.config, point.elo, point.efficiency, allocation
            )
            recommendations.append(AllocationRecommendation(
                config=point.config,
                recommended_gpu_hours=allocation,
                expected_elo_gain=expected_gain,
                priority=priority,
                rationale=rationale,
            ))

    elif strategy == "pareto":
        # Only invest in Pareto-optimal configs
        optimal_points = [p for p in pareto_points if p.is_pareto_optimal]
        if not optimal_points:
            optimal_points = pareto_points  # Fallback

        budget_per_config = total_budget / len(optimal_points)
        for point in optimal_points:
            expected_gain = estimate_marginal_elo_gain(
                point.config, point.elo, point.efficiency, budget_per_config
            )
            recommendations.append(AllocationRecommendation(
                config=point.config,
                recommended_gpu_hours=budget_per_config,
                expected_elo_gain=expected_gain,
                priority="high",
                rationale="Pareto-optimal configuration",
            ))

    return sorted(recommendations, key=lambda r: -r.expected_elo_gain)


def print_pareto_analysis(pareto_points: List[ParetoPoint]):
    """Print Pareto frontier analysis."""
    print("\n" + "=" * 70)
    print("PARETO FRONTIER ANALYSIS")
    print("=" * 70)

    print("\nPareto Frontier Points (sorted by Elo):")
    print("-" * 70)
    print(f"{'Config':<15} {'Elo':>8} {'GPU Hours':>12} {'Elo/GPU-hr':>12} {'Optimal':>10}")
    print("-" * 70)

    for point in pareto_points:
        optimal_str = "Yes" if point.is_pareto_optimal else f"No ({point.dominated_by[0]})"
        print(
            f"{point.config:<15} {point.elo:>8.1f} {point.gpu_hours:>12.2f} "
            f"{point.efficiency:>12.2f} {optimal_str:>10}"
        )

    # Summary statistics
    optimal_count = sum(1 for p in pareto_points if p.is_pareto_optimal)
    avg_efficiency = np.mean([p.efficiency for p in pareto_points])
    max_elo = max(p.elo for p in pareto_points)

    print("\nSummary:")
    print(f"  Total configs: {len(pareto_points)}")
    print(f"  Pareto optimal: {optimal_count}")
    print(f"  Best Elo: {max_elo:.1f}")
    print(f"  Avg efficiency: {avg_efficiency:.2f} Elo/GPU-hr")


def print_recommendations(
    recommendations: List[AllocationRecommendation],
    total_budget: float,
):
    """Print allocation recommendations."""
    print("\n" + "=" * 70)
    print("RESOURCE ALLOCATION RECOMMENDATIONS")
    print(f"Total Budget: {total_budget:.1f} GPU-hours")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'Config':<15} {'GPU Hours':>12} {'Expected Elo':>12} {'Priority':>10}")
    print("-" * 70)

    total_allocated = 0
    total_expected_gain = 0

    for rec in recommendations:
        print(
            f"{rec.config:<15} {rec.recommended_gpu_hours:>12.2f} "
            f"{rec.expected_elo_gain:>+12.1f} {rec.priority:>10}"
        )
        total_allocated += rec.recommended_gpu_hours
        total_expected_gain += rec.expected_elo_gain

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_allocated:>12.2f} {total_expected_gain:>+12.1f}")

    # Print detailed rationale
    print("\nDetailed Rationale:")
    for rec in recommendations:
        if rec.priority in ["high", "medium"]:
            print(f"  {rec.config}: {rec.rationale}")


def analyze_training_roi(db_path: Path):
    """Analyze return on investment for training."""
    metrics = load_training_history(db_path)

    if not metrics:
        logger.warning("No training history found")
        return

    print("\n" + "=" * 70)
    print("TRAINING ROI ANALYSIS")
    print("=" * 70)

    # Group by config
    config_metrics: Dict[str, List[TrainingMetrics]] = {}
    for m in metrics:
        if m.config not in config_metrics:
            config_metrics[m.config] = []
        config_metrics[m.config].append(m)

    print("\n" + "-" * 70)
    print(f"{'Config':<15} {'Runs':>6} {'Total GPU-hr':>12} {'Total Elo':>10} {'Avg Eff':>10}")
    print("-" * 70)

    for config, runs in sorted(config_metrics.items()):
        total_gpu = sum(r.gpu_hours for r in runs)
        total_elo_gain = sum(r.elo_gain for r in runs)
        avg_efficiency = total_elo_gain / max(total_gpu, 0.1)

        print(
            f"{config:<15} {len(runs):>6} {total_gpu:>12.2f} "
            f"{total_elo_gain:>+10.1f} {avg_efficiency:>10.2f}"
        )

    # Cost analysis
    total_gpu_hours = sum(m.gpu_hours for m in metrics)
    estimated_cost = total_gpu_hours * GPU_COST_PER_HOUR_USD["default"]
    total_elo_improvement = sum(m.elo_gain for m in metrics)

    print("\nCost Summary:")
    print(f"  Total GPU hours: {total_gpu_hours:.2f}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  Total Elo gained: {total_elo_improvement:+.1f}")
    print(f"  Cost per Elo point: ${estimated_cost / max(total_elo_improvement, 1):.2f}")


def run_optimization(
    db_path: Path,
    configs: List[str],
    budget: float,
    strategy: str = "efficiency",
) -> OptimizationResult:
    """Run multi-objective optimization."""
    logger.info("Running multi-objective optimization...")

    metrics = load_training_history(db_path)

    # Filter to specified configs if provided
    if configs:
        metrics = [m for m in metrics if m.config in configs]

    pareto_points = compute_pareto_frontier(metrics)
    recommendations = allocate_budget(pareto_points, budget, strategy)

    # Print analysis
    print_pareto_analysis(pareto_points)
    print_recommendations(recommendations, budget)

    # Create result
    result = OptimizationResult(
        pareto_frontier=[asdict(p) for p in pareto_points],
        recommendations=[asdict(r) for r in recommendations],
        total_budget=budget,
        allocated_budget=sum(r.recommended_gpu_hours for r in recommendations),
        expected_total_elo_gain=sum(r.expected_elo_gain for r in recommendations),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective optimization for RingRift AI training"
    )

    parser.add_argument(
        "--db",
        type=str,
        default=str(AI_SERVICE_ROOT / "logs" / "improvement_daemon" / "state.db"),
        help="Training history database path",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze training ROI",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Get allocation recommendations",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run full optimization",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Total GPU budget in hours",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Configs to include (e.g., square8_2p square8_3p)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["balanced", "efficiency", "underperforming", "pareto"],
        default="efficiency",
        help="Allocation strategy",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    db_path = Path(args.db)

    if args.analyze:
        analyze_training_roi(db_path)
        return 0

    if args.recommend or args.optimize:
        result = run_optimization(
            db_path=db_path,
            configs=args.configs or [],
            budget=args.budget,
            strategy=args.strategy,
        )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return 0

    # Default: show analysis and recommendations
    analyze_training_roi(db_path)
    run_optimization(
        db_path=db_path,
        configs=args.configs or [],
        budget=args.budget,
        strategy=args.strategy,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
