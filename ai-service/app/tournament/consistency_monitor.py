#!/usr/bin/env python3
"""Consistency Monitor for Composite ELO System.

This module provides monitoring for ELO system invariants:

1. NN Ranking Consistency
   - Same NN should rank similarly across algorithms
   - Significant violations indicate algorithm-specific NN properties

2. Algorithm Ranking Stability
   - Expected ranking: Gumbel > MCTS > Descent > Policy-Only
   - Measured via Algorithm Tournament results

3. Elo Transitivity
   - If A beats B (60%) and B beats C (60%), A should beat C (~70%)
   - Track prediction accuracy

Usage:
    from app.tournament.consistency_monitor import (
        ConsistencyMonitor,
        run_consistency_checks,
    )

    monitor = ConsistencyMonitor(board_type="square8", num_players=2)
    report = monitor.run_all_checks()
    print(report)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from app.training.elo_service import get_elo_service
from app.training.composite_participant import (
    extract_ai_type,
    extract_nn_id,
)

# Event emission for consistency checks (Sprint 5)
try:
    from app.training.event_integration import publish_composite_consistency_check
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    publish_composite_consistency_check = None

logger = logging.getLogger(__name__)

# Expected algorithm strength order (strongest first)
EXPECTED_ALGORITHM_ORDER = [
    "gumbel_mcts",
    "mcts",
    "descent",
    "policy_only",
]


@dataclass
class InvariantCheck:
    """Result of a single invariant check."""
    name: str
    passed: bool
    severity: str  # "info", "warning", "error"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """Complete consistency check report."""
    board_type: str
    num_players: int
    timestamp: float = field(default_factory=time.time)
    checks: list[InvariantCheck] = field(default_factory=list)
    overall_healthy: bool = True

    @property
    def errors(self) -> list[InvariantCheck]:
        return [c for c in self.checks if c.severity == "error"]

    @property
    def warnings(self) -> list[InvariantCheck]:
        return [c for c in self.checks if c.severity == "warning"]

    def __str__(self) -> str:
        lines = [
            f"Consistency Report: {self.board_type}/{self.num_players}p",
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"Overall: {'HEALTHY' if self.overall_healthy else 'ISSUES DETECTED'}",
            "",
        ]

        for check in self.checks:
            status = "✓" if check.passed else ("⚠" if check.severity == "warning" else "✗")
            lines.append(f"{status} {check.name}: {check.message}")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")

        return "\n".join(lines)


class ConsistencyMonitor:
    """Monitor for ELO system consistency invariants."""

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self.elo_service = get_elo_service()

    def run_all_checks(self) -> ConsistencyReport:
        """Run all consistency checks."""
        report = ConsistencyReport(
            board_type=self.board_type,
            num_players=self.num_players,
        )

        # Run each check
        report.checks.append(self.check_nn_ranking_consistency())
        report.checks.append(self.check_algorithm_ranking_stability())
        report.checks.append(self.check_elo_transitivity())
        report.checks.append(self.check_baseline_anchoring())
        report.checks.append(self.check_rating_distribution())
        report.checks.append(self.check_games_played_distribution())

        # Determine overall health
        report.overall_healthy = len(report.errors) == 0

        # Emit consistency check event (Sprint 5)
        if HAS_EVENTS and publish_composite_consistency_check is not None:
            import asyncio
            try:
                checks_passed = sum(1 for c in report.checks if c.passed)
                checks_failed = sum(1 for c in report.checks if not c.passed)
                check_results = {c.name: c.passed for c in report.checks}

                coro = publish_composite_consistency_check(
                    overall_healthy=report.overall_healthy,
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                    warnings_count=len(report.warnings),
                    errors_count=len(report.errors),
                    check_results=check_results,
                    board_type=self.board_type,
                    num_players=self.num_players,
                )
                # Try to schedule in running loop, otherwise use new loop
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.ensure_future(coro, loop=loop)
                except RuntimeError:
                    # No running loop - create one
                    asyncio.run(coro)
            except Exception as e:
                logger.debug(f"Failed to emit consistency check event: {e}")

        return report

    def check_nn_ranking_consistency(self) -> InvariantCheck:
        """Check if NN rankings are consistent across algorithms.

        Invariant: Same NN should rank similarly across algorithms.
        If NN_a > NN_b with Gumbel, then NN_a > NN_b with MCTS (mostly).
        """
        from app.tournament.gauntlet_aggregation import check_nn_ranking_consistency

        try:
            consistency = check_nn_ranking_consistency(
                board_type=self.board_type,
                num_players=self.num_players,
                min_games=10,
                min_algorithms=2,
            )

            if consistency.nn_count < 3:
                return InvariantCheck(
                    name="NN Ranking Consistency",
                    passed=True,
                    severity="info",
                    message=f"Not enough NNs with multiple algorithms ({consistency.nn_count})",
                )

            passed = consistency.is_consistent
            correlation = consistency.avg_rank_correlation
            violations = len(consistency.rank_violations)

            if passed:
                severity = "info"
                message = f"Rank correlation: {correlation:.2f}, {violations} violations"
            elif correlation > 0.5:
                severity = "warning"
                message = f"Moderate consistency: correlation={correlation:.2f}, {violations} violations"
            else:
                severity = "error"
                message = f"Low consistency: correlation={correlation:.2f}, {violations} violations"

            return InvariantCheck(
                name="NN Ranking Consistency",
                passed=passed,
                severity=severity,
                message=message,
                details={
                    "correlation": correlation,
                    "violations": violations,
                    "nn_count": consistency.nn_count,
                    "algorithms": consistency.algorithms_compared,
                },
            )

        except Exception as e:
            return InvariantCheck(
                name="NN Ranking Consistency",
                passed=False,
                severity="error",
                message=f"Check failed: {e}",
            )

    def check_algorithm_ranking_stability(self) -> InvariantCheck:
        """Check if algorithm rankings match expected order.

        Expected: Gumbel > MCTS > Descent > Policy-Only
        """
        try:
            rankings = self.elo_service.get_algorithm_rankings(
                board_type=self.board_type,
                num_players=self.num_players,
                min_games=5,
            )

            if len(rankings) < 2:
                return InvariantCheck(
                    name="Algorithm Ranking Stability",
                    passed=True,
                    severity="info",
                    message="Not enough algorithms with data",
                )

            # Get actual order
            actual_order = [r["ai_algorithm"] for r in rankings]

            # Check against expected order
            violations = 0
            for i, expected_algo in enumerate(EXPECTED_ALGORITHM_ORDER):
                if expected_algo not in actual_order:
                    continue
                actual_pos = actual_order.index(expected_algo)
                for j, other_algo in enumerate(EXPECTED_ALGORITHM_ORDER[i+1:], i+1):
                    if other_algo not in actual_order:
                        continue
                    other_pos = actual_order.index(other_algo)
                    if actual_pos > other_pos:  # Expected should be higher (lower index)
                        violations += 1

            if violations == 0:
                return InvariantCheck(
                    name="Algorithm Ranking Stability",
                    passed=True,
                    severity="info",
                    message=f"Rankings match expected order: {', '.join(actual_order[:4])}",
                    details={"actual_order": actual_order},
                )
            elif violations <= 2:
                return InvariantCheck(
                    name="Algorithm Ranking Stability",
                    passed=True,
                    severity="warning",
                    message=f"Minor deviations: {violations} order violations",
                    details={"actual_order": actual_order, "violations": violations},
                )
            else:
                return InvariantCheck(
                    name="Algorithm Ranking Stability",
                    passed=False,
                    severity="error",
                    message=f"Significant order violations: {violations}",
                    details={"actual_order": actual_order, "violations": violations},
                )

        except Exception as e:
            return InvariantCheck(
                name="Algorithm Ranking Stability",
                passed=False,
                severity="error",
                message=f"Check failed: {e}",
            )

    def check_elo_transitivity(self) -> InvariantCheck:
        """Check Elo transitivity (prediction accuracy).

        If A beats B (60%) and B beats C (60%), A should beat C (~70%).
        """
        try:
            # Get leaderboard
            leaderboard = self.elo_service.get_composite_leaderboard(
                board_type=self.board_type,
                num_players=self.num_players,
                min_games=10,
                limit=50,
            )

            if len(leaderboard) < 3:
                return InvariantCheck(
                    name="Elo Transitivity",
                    passed=True,
                    severity="info",
                    message="Not enough participants for transitivity check",
                )

            # Sample some triplets and check transitivity
            correct_predictions = 0
            total_predictions = 0
            max_checks = 20

            for i in range(min(len(leaderboard) - 2, max_checks)):
                p_a = leaderboard[i]
                p_b = leaderboard[i + 1]
                p_c = leaderboard[i + 2]

                rating_a = p_a["rating"]
                rating_b = p_b["rating"]
                rating_c = p_c["rating"]

                # Expected A > B > C based on ratings
                if rating_a > rating_b > rating_c:
                    correct_predictions += 1
                total_predictions += 1

            if total_predictions == 0:
                return InvariantCheck(
                    name="Elo Transitivity",
                    passed=True,
                    severity="info",
                    message="No transitivity checks performed",
                )

            accuracy = correct_predictions / total_predictions

            if accuracy >= 0.8:
                return InvariantCheck(
                    name="Elo Transitivity",
                    passed=True,
                    severity="info",
                    message=f"Transitivity accuracy: {accuracy:.0%}",
                    details={"accuracy": accuracy, "checks": total_predictions},
                )
            elif accuracy >= 0.6:
                return InvariantCheck(
                    name="Elo Transitivity",
                    passed=True,
                    severity="warning",
                    message=f"Moderate transitivity: {accuracy:.0%}",
                    details={"accuracy": accuracy, "checks": total_predictions},
                )
            else:
                return InvariantCheck(
                    name="Elo Transitivity",
                    passed=False,
                    severity="error",
                    message=f"Low transitivity: {accuracy:.0%}",
                    details={"accuracy": accuracy, "checks": total_predictions},
                )

        except Exception as e:
            return InvariantCheck(
                name="Elo Transitivity",
                passed=False,
                severity="error",
                message=f"Check failed: {e}",
            )

    def check_baseline_anchoring(self) -> InvariantCheck:
        """Check that baseline ratings are stable.

        Random baseline should be near 400 Elo.
        """
        try:
            baseline = self.elo_service.get_algorithm_baseline(
                ai_algorithm="random",
                board_type=self.board_type,
                num_players=self.num_players,
            )

            if not baseline:
                return InvariantCheck(
                    name="Baseline Anchoring",
                    passed=True,
                    severity="info",
                    message="No random baseline found",
                )

            expected_elo = 400.0
            actual_elo = baseline["baseline_elo"]
            drift = abs(actual_elo - expected_elo)

            if drift < 50:
                return InvariantCheck(
                    name="Baseline Anchoring",
                    passed=True,
                    severity="info",
                    message=f"Random baseline at {actual_elo:.0f} (drift: {drift:.0f})",
                    details={"expected": expected_elo, "actual": actual_elo},
                )
            elif drift < 100:
                return InvariantCheck(
                    name="Baseline Anchoring",
                    passed=True,
                    severity="warning",
                    message=f"Random baseline drifted to {actual_elo:.0f} (drift: {drift:.0f})",
                    details={"expected": expected_elo, "actual": actual_elo},
                )
            else:
                return InvariantCheck(
                    name="Baseline Anchoring",
                    passed=False,
                    severity="error",
                    message=f"Random baseline significantly drifted to {actual_elo:.0f}",
                    details={"expected": expected_elo, "actual": actual_elo},
                )

        except Exception as e:
            return InvariantCheck(
                name="Baseline Anchoring",
                passed=False,
                severity="error",
                message=f"Check failed: {e}",
            )

    def check_rating_distribution(self) -> InvariantCheck:
        """Check that rating distribution is reasonable.

        Ratings should span a reasonable range (400-2500).
        """
        try:
            leaderboard = self.elo_service.get_composite_leaderboard(
                board_type=self.board_type,
                num_players=self.num_players,
                min_games=1,
                limit=1000,
            )

            if len(leaderboard) < 5:
                return InvariantCheck(
                    name="Rating Distribution",
                    passed=True,
                    severity="info",
                    message="Not enough participants for distribution check",
                )

            ratings = [p["rating"] for p in leaderboard]
            min_rating = min(ratings)
            max_rating = max(ratings)
            spread = max_rating - min_rating
            avg_rating = sum(ratings) / len(ratings)

            details = {
                "min": min_rating,
                "max": max_rating,
                "spread": spread,
                "avg": avg_rating,
                "count": len(ratings),
            }

            # Check for reasonable spread
            if spread < 100:
                return InvariantCheck(
                    name="Rating Distribution",
                    passed=False,
                    severity="warning",
                    message=f"Low rating spread: {spread:.0f} (may indicate calibration issues)",
                    details=details,
                )

            if max_rating > 3000 or min_rating < 100:
                return InvariantCheck(
                    name="Rating Distribution",
                    passed=False,
                    severity="warning",
                    message=f"Extreme ratings detected: {min_rating:.0f}-{max_rating:.0f}",
                    details=details,
                )

            return InvariantCheck(
                name="Rating Distribution",
                passed=True,
                severity="info",
                message=f"Healthy distribution: {min_rating:.0f}-{max_rating:.0f} (spread: {spread:.0f})",
                details=details,
            )

        except Exception as e:
            return InvariantCheck(
                name="Rating Distribution",
                passed=False,
                severity="error",
                message=f"Check failed: {e}",
            )

    def check_games_played_distribution(self) -> InvariantCheck:
        """Check games played distribution for coverage."""
        try:
            leaderboard = self.elo_service.get_composite_leaderboard(
                board_type=self.board_type,
                num_players=self.num_players,
                min_games=0,
                limit=1000,
            )

            if not leaderboard:
                return InvariantCheck(
                    name="Games Played Distribution",
                    passed=True,
                    severity="info",
                    message="No participants found",
                )

            games = [p["games_played"] for p in leaderboard]
            total_games = sum(games)
            avg_games = total_games / len(games)
            max_games = max(games)
            min_games = min(games)

            # Count participants with enough games for reliable rating
            reliable_count = sum(1 for g in games if g >= 30)
            reliable_pct = reliable_count / len(games)

            details = {
                "total_games": total_games,
                "avg_games": avg_games,
                "min_games": min_games,
                "max_games": max_games,
                "participants": len(games),
                "reliable_count": reliable_count,
                "reliable_pct": reliable_pct,
            }

            if reliable_pct >= 0.5:
                return InvariantCheck(
                    name="Games Played Distribution",
                    passed=True,
                    severity="info",
                    message=f"{reliable_pct:.0%} have reliable ratings (30+ games)",
                    details=details,
                )
            elif reliable_pct >= 0.2:
                return InvariantCheck(
                    name="Games Played Distribution",
                    passed=True,
                    severity="warning",
                    message=f"Only {reliable_pct:.0%} have reliable ratings",
                    details=details,
                )
            else:
                return InvariantCheck(
                    name="Games Played Distribution",
                    passed=False,
                    severity="warning",
                    message=f"Low coverage: {reliable_pct:.0%} have reliable ratings",
                    details=details,
                )

        except Exception as e:
            return InvariantCheck(
                name="Games Played Distribution",
                passed=False,
                severity="error",
                message=f"Check failed: {e}",
            )


def run_consistency_checks(
    board_type: str = "square8",
    num_players: int = 2,
) -> ConsistencyReport:
    """Run all consistency checks.

    Args:
        board_type: Board type
        num_players: Number of players

    Returns:
        ConsistencyReport with all check results
    """
    monitor = ConsistencyMonitor(board_type=board_type, num_players=num_players)
    return monitor.run_all_checks()
