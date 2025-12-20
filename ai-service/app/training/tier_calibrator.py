"""
Automated tier threshold calibration via A/B testing.

This module replaces manual tier calibration experiments with automated
A/B testing that uses statistical significance to auto-promote tier thresholds.
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CalibrationStatus(str, Enum):
    """Status of a calibration proposal."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CalibrationConfig:
    """Configuration for tier calibration."""

    # Test parameters
    min_games_for_significance: int = 200
    significance_threshold: float = 0.95  # 95% confidence
    max_concurrent_calibrations: int = 2

    # Threshold bounds
    min_elo_threshold: float = 800
    max_elo_threshold: float = 2400
    max_threshold_change_percent: float = 10.0  # Max 10% change per calibration

    # Timing
    calibration_timeout_hours: int = 72  # 3 days max per test
    min_time_between_calibrations_hours: int = 24

    # Auto-calibration triggers
    auto_calibrate: bool = True
    trigger_on_promotion_failure_rate: float = 0.3  # Trigger if > 30% fail gate
    trigger_on_win_rate_deviation: float = 0.15     # Trigger if win rate off by > 15%


@dataclass
class CalibrationProposal:
    """A proposed tier calibration to test."""
    proposal_id: str
    tier: str
    control_threshold: float
    treatment_threshold: float
    required_games: int
    significance_level: float
    status: CalibrationStatus = CalibrationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    reason: str = ""


@dataclass
class CalibrationResult:
    """Result of a completed calibration test."""
    proposal_id: str
    tier: str
    control_threshold: float
    treatment_threshold: float
    control_games: int
    treatment_games: int
    control_win_rate: float
    treatment_win_rate: float
    p_value: float
    is_significant: bool
    treatment_wins: bool
    confidence_interval: tuple  # (lower, upper)
    recommendation: str


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    tier: str
    control_threshold: float
    treatment_threshold: float
    required_games: int
    significance_level: float
    started_at: float = field(default_factory=time.time)


@dataclass
class TierThreshold:
    """Current threshold for a tier."""
    tier: str
    elo_threshold: float
    last_calibrated: float
    calibration_count: int = 0


class TierCalibrator:
    """
    Automated tier threshold calibration via A/B testing.

    Monitors tier performance and proposes calibration tests when
    thresholds appear miscalibrated. Uses statistical significance
    testing to auto-apply successful calibrations.
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()

        # State
        self._proposals: dict[str, CalibrationProposal] = {}
        self._thresholds: dict[str, TierThreshold] = {}
        self._results: list[CalibrationResult] = []

        # Tier performance tracking
        self._tier_stats: dict[str, dict[str, Any]] = {}

        # Initialize default tiers
        self._init_default_tiers()

    def _init_default_tiers(self) -> None:
        """Initialize default tier thresholds."""
        default_tiers = {
            "D2": 1000,
            "D4": 1200,
            "D6": 1400,
            "D8": 1600,
            "D10": 1800,
        }
        for tier, threshold in default_tiers.items():
            self._thresholds[tier] = TierThreshold(
                tier=tier,
                elo_threshold=threshold,
                last_calibrated=0,
            )

    def get_current_threshold(self, tier: str) -> float:
        """Get current Elo threshold for a tier."""
        if tier in self._thresholds:
            return self._thresholds[tier].elo_threshold
        return 1000.0  # Default

    def set_threshold(self, tier: str, threshold: float) -> None:
        """Set the threshold for a tier."""
        if tier not in self._thresholds:
            self._thresholds[tier] = TierThreshold(
                tier=tier,
                elo_threshold=threshold,
                last_calibrated=time.time(),
            )
        else:
            self._thresholds[tier].elo_threshold = threshold
            self._thresholds[tier].last_calibrated = time.time()
            self._thresholds[tier].calibration_count += 1

    def propose_calibration(
        self,
        tier: str,
        proposed_threshold: float,
        reason: str = "auto_proposed",
    ) -> CalibrationProposal | None:
        """
        Create a calibration proposal for testing.

        Args:
            tier: The tier to calibrate
            proposed_threshold: The proposed new threshold
            reason: Reason for the proposal

        Returns:
            CalibrationProposal if valid, None if rejected
        """
        current = self.get_current_threshold(tier)

        # Validate bounds
        if proposed_threshold < self.config.min_elo_threshold:
            logger.warning(f"Proposed threshold {proposed_threshold} below minimum")
            return None

        if proposed_threshold > self.config.max_elo_threshold:
            logger.warning(f"Proposed threshold {proposed_threshold} above maximum")
            return None

        # Check max change
        change_percent = abs(proposed_threshold - current) / current * 100
        if change_percent > self.config.max_threshold_change_percent:
            logger.warning(
                f"Proposed change {change_percent:.1f}% exceeds max "
                f"{self.config.max_threshold_change_percent}%"
            )
            return None

        # Check concurrent calibrations
        active = sum(
            1 for p in self._proposals.values()
            if p.status in (CalibrationStatus.PENDING, CalibrationStatus.RUNNING)
        )
        if active >= self.config.max_concurrent_calibrations:
            logger.warning(f"Max concurrent calibrations reached ({active})")
            return None

        # Check time since last calibration
        if tier in self._thresholds:
            hours_since = (time.time() - self._thresholds[tier].last_calibrated) / 3600
            if hours_since < self.config.min_time_between_calibrations_hours:
                logger.info(
                    f"Too soon to calibrate {tier} ({hours_since:.1f}h < "
                    f"{self.config.min_time_between_calibrations_hours}h)"
                )
                return None

        proposal = CalibrationProposal(
            proposal_id=str(uuid.uuid4())[:8],
            tier=tier,
            control_threshold=current,
            treatment_threshold=proposed_threshold,
            required_games=self.config.min_games_for_significance,
            significance_level=self.config.significance_threshold,
            reason=reason,
        )

        self._proposals[proposal.proposal_id] = proposal
        logger.info(
            f"Created calibration proposal {proposal.proposal_id} for {tier}: "
            f"{current} -> {proposed_threshold} ({reason})"
        )

        return proposal

    def get_pending_calibrations(self) -> list[CalibrationProposal]:
        """Get list of pending calibration proposals."""
        return [
            p for p in self._proposals.values()
            if p.status == CalibrationStatus.PENDING
        ]

    def get_running_calibrations(self) -> list[CalibrationProposal]:
        """Get list of running calibration proposals."""
        return [
            p for p in self._proposals.values()
            if p.status == CalibrationStatus.RUNNING
        ]

    def start_calibration(self, proposal_id: str) -> ABTestConfig | None:
        """
        Start a calibration test.

        Returns ABTestConfig for the A/B test manager.
        """
        if proposal_id not in self._proposals:
            return None

        proposal = self._proposals[proposal_id]
        if proposal.status != CalibrationStatus.PENDING:
            return None

        proposal.status = CalibrationStatus.RUNNING
        proposal.started_at = time.time()

        return ABTestConfig(
            test_id=f"tier_cal_{proposal.proposal_id}",
            tier=proposal.tier,
            control_threshold=proposal.control_threshold,
            treatment_threshold=proposal.treatment_threshold,
            required_games=proposal.required_games,
            significance_level=proposal.significance_level,
        )

    def complete_calibration(
        self,
        proposal_id: str,
        control_games: int,
        treatment_games: int,
        control_win_rate: float,
        treatment_win_rate: float,
    ) -> CalibrationResult:
        """
        Complete a calibration with test results.

        Args:
            proposal_id: The proposal being completed
            control_games: Games played with control threshold
            treatment_games: Games played with treatment threshold
            control_win_rate: Win rate with control
            treatment_win_rate: Win rate with treatment

        Returns:
            CalibrationResult with analysis
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        # Calculate statistical significance
        p_value, is_significant, ci = self._calculate_significance(
            control_games, treatment_games,
            control_win_rate, treatment_win_rate,
            proposal.significance_level,
        )

        # Determine if treatment is better
        treatment_wins = (
            is_significant and
            treatment_win_rate > control_win_rate
        )

        # Generate recommendation
        if not is_significant:
            recommendation = "no_change_insufficient_significance"
        elif treatment_wins:
            recommendation = f"adopt_treatment_{proposal.treatment_threshold}"
        else:
            recommendation = "keep_control"

        result = CalibrationResult(
            proposal_id=proposal_id,
            tier=proposal.tier,
            control_threshold=proposal.control_threshold,
            treatment_threshold=proposal.treatment_threshold,
            control_games=control_games,
            treatment_games=treatment_games,
            control_win_rate=control_win_rate,
            treatment_win_rate=treatment_win_rate,
            p_value=p_value,
            is_significant=is_significant,
            treatment_wins=treatment_wins,
            confidence_interval=ci,
            recommendation=recommendation,
        )

        # Update proposal status
        proposal.status = CalibrationStatus.COMPLETED
        proposal.completed_at = time.time()

        # Store result
        self._results.append(result)

        logger.info(
            f"Completed calibration {proposal_id}: significant={is_significant}, "
            f"treatment_wins={treatment_wins}, p={p_value:.4f}"
        )

        return result

    def apply_calibration(self, result: CalibrationResult) -> bool:
        """
        Apply a calibration result if successful.

        Returns True if calibration was applied.
        """
        if not result.is_significant or not result.treatment_wins:
            logger.info(f"Not applying calibration {result.proposal_id}: not significant or control better")
            return False

        old_threshold = self.get_current_threshold(result.tier)
        self.set_threshold(result.tier, result.treatment_threshold)

        logger.info(
            f"Applied calibration for {result.tier}: "
            f"{old_threshold} -> {result.treatment_threshold}"
        )

        return True

    def _calculate_significance(
        self,
        n1: int,
        n2: int,
        p1: float,
        p2: float,
        alpha: float,
    ) -> tuple:
        """
        Calculate statistical significance using two-proportion z-test.

        Returns (p_value, is_significant, confidence_interval)
        """
        if n1 < 10 or n2 < 10:
            return (1.0, False, (0.0, 0.0))

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se == 0:
            return (1.0, False, (0.0, 0.0))

        # Z-score
        z = (p2 - p1) / se

        # Two-tailed p-value (approximation using error function)
        p_value = 2 * (1 - self._norm_cdf(abs(z)))

        is_significant = p_value < (1 - alpha)

        # Confidence interval for difference
        z_crit = 1.96  # 95% CI
        diff = p2 - p1
        se_diff = math.sqrt(p1 * (1-p1) / n1 + p2 * (1-p2) / n2)
        ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

        return (p_value, is_significant, ci)

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def update_tier_stats(
        self,
        tier: str,
        promotion_attempts: int,
        promotion_successes: int,
        avg_win_rate: float,
        expected_win_rate: float = 0.5,
    ) -> None:
        """
        Update tier performance statistics.

        Used for auto-calibration trigger detection.
        """
        self._tier_stats[tier] = {
            "promotion_attempts": promotion_attempts,
            "promotion_successes": promotion_successes,
            "promotion_rate": promotion_successes / max(1, promotion_attempts),
            "avg_win_rate": avg_win_rate,
            "expected_win_rate": expected_win_rate,
            "win_rate_deviation": abs(avg_win_rate - expected_win_rate),
            "updated_at": time.time(),
        }

    def check_auto_calibration_triggers(self) -> list[CalibrationProposal]:
        """
        Check if any tiers need auto-calibration.

        Returns list of proposed calibrations based on performance data.
        """
        if not self.config.auto_calibrate:
            return []

        proposals = []

        for tier, stats in self._tier_stats.items():
            # Check promotion failure rate
            failure_rate = 1 - stats.get("promotion_rate", 1.0)
            if failure_rate > self.config.trigger_on_promotion_failure_rate:
                # High failure rate suggests threshold is too high
                current = self.get_current_threshold(tier)
                proposed = current * 0.95  # Lower by 5%

                proposal = self.propose_calibration(
                    tier=tier,
                    proposed_threshold=proposed,
                    reason=f"high_failure_rate_{failure_rate:.2f}",
                )
                if proposal:
                    proposals.append(proposal)

            # Check win rate deviation
            deviation = stats.get("win_rate_deviation", 0)
            avg_win_rate = stats.get("avg_win_rate", 0.5)

            if deviation > self.config.trigger_on_win_rate_deviation:
                current = self.get_current_threshold(tier)

                # If win rate too high, threshold might be too low
                if avg_win_rate > 0.5 + self.config.trigger_on_win_rate_deviation:
                    proposed = current * 1.05  # Raise by 5%
                    reason = f"high_win_rate_{avg_win_rate:.2f}"
                else:
                    proposed = current * 0.95  # Lower by 5%
                    reason = f"low_win_rate_{avg_win_rate:.2f}"

                proposal = self.propose_calibration(
                    tier=tier,
                    proposed_threshold=proposed,
                    reason=reason,
                )
                if proposal:
                    proposals.append(proposal)

        return proposals

    def get_calibration_stats(self) -> dict[str, Any]:
        """Get calibration statistics for monitoring."""
        recent_results = [
            r for r in self._results
            if time.time() - self._proposals.get(r.proposal_id, CalibrationProposal(
                proposal_id="", tier="", control_threshold=0,
                treatment_threshold=0, required_games=0, significance_level=0,
                completed_at=0
            )).completed_at < 86400 * 7  # Last 7 days
        ]

        successful = sum(1 for r in recent_results if r.treatment_wins)
        failed = len(recent_results) - successful

        return {
            "auto_calibrate_enabled": self.config.auto_calibrate,
            "pending_proposals": len(self.get_pending_calibrations()),
            "running_calibrations": len(self.get_running_calibrations()),
            "completed_last_week": len(recent_results),
            "successful_last_week": successful,
            "failed_last_week": failed,
            "current_thresholds": {
                tier: t.elo_threshold
                for tier, t in self._thresholds.items()
            },
            "tier_stats": self._tier_stats,
        }


def load_calibration_config_from_yaml(yaml_config: dict[str, Any]) -> CalibrationConfig:
    """Load CalibrationConfig from YAML configuration dict."""
    tier_cal = yaml_config.get("tier_calibration", {})

    return CalibrationConfig(
        auto_calibrate=tier_cal.get("auto_calibrate", True),
        min_games_for_significance=tier_cal.get("min_games_for_significance", 200),
        significance_threshold=tier_cal.get("significance_threshold", 0.95),
        max_concurrent_calibrations=tier_cal.get("max_concurrent_calibrations", 2),
        min_elo_threshold=tier_cal.get("min_elo_threshold", 800),
        max_elo_threshold=tier_cal.get("max_elo_threshold", 2400),
        max_threshold_change_percent=tier_cal.get("max_threshold_change_percent", 10.0),
        calibration_timeout_hours=tier_cal.get("calibration_timeout_hours", 72),
        min_time_between_calibrations_hours=tier_cal.get("min_time_between_calibrations_hours", 24),
        trigger_on_promotion_failure_rate=tier_cal.get("trigger_on_promotion_failure_rate", 0.3),
        trigger_on_win_rate_deviation=tier_cal.get("trigger_on_win_rate_deviation", 0.15),
    )
