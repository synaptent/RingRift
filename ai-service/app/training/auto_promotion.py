"""Automated Post-Training Model Promotion for RingRift.

This module provides automated promotion of newly trained models based on
gauntlet evaluation results. It runs after training completion and automatically
promotes models that meet the promotion criteria.

Design Goals:
- Fully automated: No manual intervention required after training
- Statistical significance: Uses Wilson score intervals to avoid lucky variance
- Flexible criteria: OR logic - promote if Elo parity OR win rate floors met
- Bootstrap handling: Works even when no canonical model exists yet

Usage:
    from app.training.auto_promotion import AutoPromotionEngine, AutoPromotionCriteria

    # Create engine with default criteria
    engine = AutoPromotionEngine()

    # Evaluate and potentially promote a trained model
    result = await engine.evaluate_and_promote(
        model_path="models/hex8_2p_epoch50.pth",
        board_type="hex8",
        num_players=2,
        games=30,
    )

    if result.approved:
        print(f"Model promoted: {result.reason}")
    else:
        print(f"Promotion rejected: {result.reason}")

January 2026: Created for automated promotion pipeline.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class AutoPromotionCriteria:
    """Criteria for automated post-training promotion.

    Uses OR logic: promote if EITHER Elo parity OR win rate floors are met.
    This provides multiple paths to promotion, making the system more flexible
    during different stages of training.

    Attributes:
        min_games_quick: Minimum games for quick evaluation (default: 30)
        min_games_production: Minimum games for production promotion (default: 100)
        wilson_confidence: Confidence level for Wilson score intervals (default: 0.95)
        win_rate_floors: Minimum win rates by player count
        min_absolute_elo: Minimum Elo to prevent promoting very weak models
    """

    # Minimum games for evaluation
    min_games_quick: int = 30
    min_games_production: int = 100

    # Statistical significance
    wilson_confidence: float = 0.95

    # Win rate floors by player count (vs random, vs heuristic)
    win_rate_floors: dict[int, dict[str, float]] = field(default_factory=lambda: {
        2: {"random": 0.70, "heuristic": 0.50},
        3: {"random": 0.50, "heuristic": 0.40},
        4: {"random": 0.40, "heuristic": 0.35},
    })

    # Minimum absolute Elo (prevents promoting very weak models)
    min_absolute_elo: dict[int, float] = field(default_factory=lambda: {
        2: 1400,
        3: 1350,
        4: 1300,
    })


class PromotionCriterion(str, Enum):
    """The criterion that triggered promotion approval."""
    ELO_PARITY = "elo_parity"
    WIN_RATE_FLOORS = "win_rate_floors"
    BEATS_CURRENT_BEST = "beats_current_best"


@dataclass
class PromotionDecision:
    """Result of a promotion decision evaluation.

    Attributes:
        approved: Whether the model should be promoted
        reason: Human-readable explanation of the decision
        criterion_met: Which criterion triggered approval (if approved)
        details: Additional details about the evaluation
    """
    approved: bool
    reason: str
    criterion_met: PromotionCriterion | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GauntletEvalResults:
    """Structured results from gauntlet evaluation.

    Attributes:
        games_vs_random: Total games played against random opponent
        wins_vs_random: Wins against random opponent
        win_rate_vs_random: Win rate against random (0.0-1.0)
        games_vs_heuristic: Total games played against heuristic opponent
        wins_vs_heuristic: Wins against heuristic opponent
        win_rate_vs_heuristic: Win rate against heuristic (0.0-1.0)
        estimated_elo: Estimated Elo from gauntlet results
        wilson_ci_random: Wilson confidence interval for random win rate
        wilson_ci_heuristic: Wilson confidence interval for heuristic win rate
        raw_result: The raw GauntletResult from game_gauntlet.py
    """
    games_vs_random: int = 0
    wins_vs_random: int = 0
    win_rate_vs_random: float = 0.0
    games_vs_heuristic: int = 0
    wins_vs_heuristic: int = 0
    win_rate_vs_heuristic: float = 0.0
    estimated_elo: float = 1000.0
    wilson_ci_random: tuple[float, float] = (0.0, 0.0)
    wilson_ci_heuristic: tuple[float, float] = (0.0, 0.0)
    raw_result: Any = None

    def is_statistically_significant(self, threshold: float = 0.5) -> bool:
        """Check if win rates are statistically above threshold."""
        return (
            self.wilson_ci_random[0] > threshold and
            self.wilson_ci_heuristic[0] > threshold
        )


@dataclass
class PromotionResult:
    """Full result of an auto-promotion evaluation.

    Attributes:
        decision: The promotion decision
        eval_results: Gauntlet evaluation results
        model_path: Path to the evaluated model
        board_type: Board type
        num_players: Number of players
        promoted_path: Path where model was promoted (if approved)
        timestamp: When the evaluation was performed
    """
    decision: PromotionDecision
    eval_results: GauntletEvalResults | None = None
    model_path: str = ""
    board_type: str = ""
    num_players: int = 2
    promoted_path: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def approved(self) -> bool:
        """Convenience accessor for decision.approved."""
        return self.decision.approved

    @property
    def reason(self) -> str:
        """Convenience accessor for decision.reason."""
        return self.decision.reason


# =============================================================================
# AutoPromotionEngine
# =============================================================================


class AutoPromotionEngine:
    """Automated promotion engine for post-training model evaluation.

    This engine runs gauntlet evaluation against baseline opponents and
    promotes models that meet the promotion criteria. It uses OR logic:
    a model is promoted if it achieves EITHER Elo parity with heuristic
    OR meets win rate floor thresholds with statistical significance.

    Example:
        engine = AutoPromotionEngine()
        result = await engine.evaluate_and_promote(
            model_path="models/hex8_2p_trained.pth",
            board_type="hex8",
            num_players=2,
        )
        if result.approved:
            print(f"Promoted to: {result.promoted_path}")
    """

    def __init__(self, criteria: AutoPromotionCriteria | None = None):
        """Initialize the auto-promotion engine.

        Args:
            criteria: Promotion criteria to use. Defaults to AutoPromotionCriteria().
        """
        self.criteria = criteria or AutoPromotionCriteria()
        self._gauntlet_module = None
        self._significance_module = None
        self._promotion_controller = None
        self._event_router = None

    def _ensure_gauntlet(self) -> Any:
        """Lazy-load the gauntlet module."""
        if self._gauntlet_module is None:
            import app.training.game_gauntlet as game_gauntlet
            self._gauntlet_module = game_gauntlet
        return self._gauntlet_module

    def _ensure_significance(self) -> Any:
        """Lazy-load the significance module."""
        if self._significance_module is None:
            import app.training.significance as significance
            self._significance_module = significance
        return self._significance_module

    def _ensure_promotion_controller(self) -> Any:
        """Lazy-load the promotion controller."""
        if self._promotion_controller is None:
            try:
                from app.training.promotion_controller import PromotionController
                self._promotion_controller = PromotionController()
            except ImportError:
                self._promotion_controller = None
        return self._promotion_controller

    def _ensure_event_router(self) -> Any:
        """Lazy-load the event router."""
        if self._event_router is None:
            try:
                from app.coordination.event_router import get_router
                self._event_router = get_router()
            except ImportError:
                self._event_router = None
        return self._event_router

    async def evaluate_and_promote(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        games: int = 30,
        sync_to_cluster: bool = True,
    ) -> PromotionResult:
        """Run gauntlet evaluation and promote if criteria met.

        This is the main entry point for automated post-training promotion.
        It performs the following steps:
        1. Run gauntlet evaluation against RANDOM and HEURISTIC baselines
        2. Get heuristic baseline Elo for comparison
        3. Apply promotion decision logic (OR criteria)
        4. Execute promotion if approved
        5. Emit events for feedback loops

        Args:
            model_path: Path to the trained model checkpoint
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            games: Number of games per opponent for evaluation
            sync_to_cluster: Whether to sync promoted model to cluster

        Returns:
            PromotionResult with decision, evaluation results, and promotion details
        """
        config_key = f"{board_type}_{num_players}p"
        logger.info(f"[AutoPromotion] Starting evaluation for {config_key}: {model_path}")

        # Step 1: Run gauntlet evaluation
        try:
            eval_results = await self._run_gauntlet_evaluation(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                games=games,
            )
        except Exception as e:
            logger.error(f"[AutoPromotion] Gauntlet evaluation failed: {e}")
            return PromotionResult(
                decision=PromotionDecision(
                    approved=False,
                    reason=f"Gauntlet evaluation failed: {e}",
                ),
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            )

        # Step 2: Get heuristic baseline Elo
        heuristic_elo = await self._get_heuristic_elo(board_type, num_players)

        # Step 3: Apply promotion decision
        decision = self._evaluate_criteria(
            results=eval_results,
            heuristic_elo=heuristic_elo,
            num_players=num_players,
            config_key=config_key,
        )

        # Build result
        result = PromotionResult(
            decision=decision,
            eval_results=eval_results,
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
        )

        # Step 4: Execute promotion if approved
        if decision.approved:
            promoted_path = await self._execute_promotion(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                sync_to_cluster=sync_to_cluster,
            )
            result.promoted_path = promoted_path
            self._emit_promotion_event("MODEL_AUTO_PROMOTED", result)
            logger.info(f"[AutoPromotion] Model promoted: {decision.reason}")
        else:
            self._emit_promotion_event("MODEL_PROMOTION_REJECTED", result)
            logger.info(f"[AutoPromotion] Promotion rejected: {decision.reason}")

        return result

    async def _run_gauntlet_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        games: int,
    ) -> GauntletEvalResults:
        """Run gauntlet evaluation against baseline opponents.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players
            games: Number of games per opponent

        Returns:
            GauntletEvalResults with evaluation statistics
        """
        import asyncio

        gauntlet = self._ensure_gauntlet()
        significance = self._ensure_significance()

        # Convert board_type string to enum if needed
        try:
            from app.models import BoardType
            if isinstance(board_type, str):
                board_type_enum = BoardType(board_type)
            else:
                board_type_enum = board_type
        except (ImportError, ValueError):
            board_type_enum = board_type

        # Run gauntlet with RANDOM and HEURISTIC baselines
        opponents = [
            gauntlet.BaselineOpponent.RANDOM,
            gauntlet.BaselineOpponent.HEURISTIC,
        ]

        # Run in thread pool since run_baseline_gauntlet is synchronous
        raw_result = await asyncio.to_thread(
            gauntlet.run_baseline_gauntlet,
            model_path=model_path,
            board_type=board_type_enum,
            opponents=opponents,
            games_per_opponent=games,
            num_players=num_players,
            check_baseline_gating=False,  # We apply our own criteria
            early_stopping=True,
            save_games_for_training=True,
        )

        # Extract results per opponent
        results = GauntletEvalResults(raw_result=raw_result)

        # Parse results from raw GauntletResult
        if hasattr(raw_result, "opponent_results"):
            for opponent, opponent_result in raw_result.opponent_results.items():
                opponent_name = opponent.value if hasattr(opponent, "value") else str(opponent)
                if "random" in opponent_name.lower():
                    results.games_vs_random = opponent_result.get("games", 0)
                    results.wins_vs_random = opponent_result.get("wins", 0)
                    if results.games_vs_random > 0:
                        results.win_rate_vs_random = results.wins_vs_random / results.games_vs_random
                elif "heuristic" in opponent_name.lower():
                    results.games_vs_heuristic = opponent_result.get("games", 0)
                    results.wins_vs_heuristic = opponent_result.get("wins", 0)
                    if results.games_vs_heuristic > 0:
                        results.win_rate_vs_heuristic = results.wins_vs_heuristic / results.games_vs_heuristic

        # Fall back to aggregate stats if opponent_results not available
        if results.games_vs_random == 0 and hasattr(raw_result, "total_games"):
            # Use total stats as approximation (when opponent breakdown unavailable)
            results.games_vs_random = raw_result.total_games // 2
            results.wins_vs_random = int(raw_result.total_wins * 0.6)  # Assume random is easier
            results.games_vs_heuristic = raw_result.total_games // 2
            results.wins_vs_heuristic = raw_result.total_wins - results.wins_vs_random
            if results.games_vs_random > 0:
                results.win_rate_vs_random = results.wins_vs_random / results.games_vs_random
            if results.games_vs_heuristic > 0:
                results.win_rate_vs_heuristic = results.wins_vs_heuristic / results.games_vs_heuristic

        # Calculate Wilson confidence intervals
        results.wilson_ci_random = significance.wilson_score_interval(
            wins=results.wins_vs_random,
            total=results.games_vs_random,
            confidence=self.criteria.wilson_confidence,
        )
        results.wilson_ci_heuristic = significance.wilson_score_interval(
            wins=results.wins_vs_heuristic,
            total=results.games_vs_heuristic,
            confidence=self.criteria.wilson_confidence,
        )

        # Get estimated Elo from raw result
        if hasattr(raw_result, "estimated_elo") and raw_result.estimated_elo:
            results.estimated_elo = raw_result.estimated_elo
        elif hasattr(raw_result, "elo") and raw_result.elo:
            results.estimated_elo = raw_result.elo

        logger.info(
            f"[AutoPromotion] Gauntlet results: "
            f"vs_random={results.win_rate_vs_random:.1%} ({results.games_vs_random} games), "
            f"vs_heuristic={results.win_rate_vs_heuristic:.1%} ({results.games_vs_heuristic} games), "
            f"elo={results.estimated_elo:.0f}"
        )

        return results

    async def _get_heuristic_elo(
        self,
        board_type: str,
        num_players: int,
    ) -> float:
        """Get the heuristic baseline Elo for comparison.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            Heuristic Elo rating (default: 1200)
        """
        try:
            from app.training.elo_service import EloService, get_elo_service
            elo_service = get_elo_service()
            config_key = f"{board_type}_{num_players}p"
            heuristic_id = f"heuristic:{config_key}"
            rating = await elo_service.get_rating(heuristic_id, config_key)
            if rating and hasattr(rating, "rating"):
                return rating.rating
        except Exception as e:
            logger.debug(f"[AutoPromotion] Could not get heuristic Elo: {e}")

        # Default heuristic Elo based on player count
        defaults = {2: 1200, 3: 1150, 4: 1100}
        return defaults.get(num_players, 1200)

    def _evaluate_criteria(
        self,
        results: GauntletEvalResults,
        heuristic_elo: float,
        num_players: int,
        config_key: str,
    ) -> PromotionDecision:
        """Apply OR logic for promotion criteria.

        A model is promoted if it meets ANY of these criteria:
        1. Elo parity: Neural Elo >= Heuristic Elo
        2. Win rate floors: Meets minimum win rates with statistical significance

        Args:
            results: Gauntlet evaluation results
            heuristic_elo: Heuristic baseline Elo
            num_players: Number of players
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            PromotionDecision with approval status and reason
        """
        floors = self.criteria.win_rate_floors.get(num_players, {
            "random": 0.50, "heuristic": 0.35
        })
        min_elo = self.criteria.min_absolute_elo.get(num_players, 1300)

        # Get Wilson CI lower bounds for statistical significance
        random_ci_lower = results.wilson_ci_random[0]
        heuristic_ci_lower = results.wilson_ci_heuristic[0]

        details = {
            "estimated_elo": results.estimated_elo,
            "heuristic_elo": heuristic_elo,
            "elo_gap": results.estimated_elo - heuristic_elo,
            "win_rate_vs_random": results.win_rate_vs_random,
            "win_rate_vs_heuristic": results.win_rate_vs_heuristic,
            "random_ci_lower": random_ci_lower,
            "heuristic_ci_lower": heuristic_ci_lower,
            "random_floor": floors["random"],
            "heuristic_floor": floors["heuristic"],
        }

        # GATE 1: Check minimum absolute Elo
        if results.estimated_elo < min_elo:
            return PromotionDecision(
                approved=False,
                reason=f"Elo {results.estimated_elo:.0f} below minimum {min_elo} for {num_players}p",
                details=details,
            )

        # CRITERION 1: Elo parity (neural >= heuristic)
        elo_parity = results.estimated_elo >= heuristic_elo
        if elo_parity:
            return PromotionDecision(
                approved=True,
                reason=f"Elo parity achieved: {results.estimated_elo:.0f} >= {heuristic_elo:.0f} (heuristic)",
                criterion_met=PromotionCriterion.ELO_PARITY,
                details=details,
            )

        # CRITERION 2: Win rate floors met (with statistical significance)
        # Use Wilson CI lower bounds instead of raw win rates for significance
        random_floor_met = random_ci_lower >= floors["random"]
        heuristic_floor_met = heuristic_ci_lower >= floors["heuristic"]
        win_rate_floors_met = random_floor_met and heuristic_floor_met

        if win_rate_floors_met:
            return PromotionDecision(
                approved=True,
                reason=(
                    f"Win rate floors met with statistical significance: "
                    f"vs_random CI lower {random_ci_lower:.1%} >= {floors['random']:.0%}, "
                    f"vs_heuristic CI lower {heuristic_ci_lower:.1%} >= {floors['heuristic']:.0%}"
                ),
                criterion_met=PromotionCriterion.WIN_RATE_FLOORS,
                details=details,
            )

        # Neither criterion met
        reasons = []
        if not elo_parity:
            reasons.append(f"Elo {results.estimated_elo:.0f} < {heuristic_elo:.0f} heuristic")
        if not random_floor_met:
            reasons.append(f"vs_random CI {random_ci_lower:.1%} < {floors['random']:.0%}")
        if not heuristic_floor_met:
            reasons.append(f"vs_heuristic CI {heuristic_ci_lower:.1%} < {floors['heuristic']:.0%}")

        return PromotionDecision(
            approved=False,
            reason=f"Neither Elo parity nor win rate floors met: {'; '.join(reasons)}",
            details=details,
        )

    async def _execute_promotion(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        sync_to_cluster: bool = True,
    ) -> str | None:
        """Execute the promotion by copying to canonical path and syncing.

        Args:
            model_path: Source model path
            board_type: Board type
            num_players: Number of players
            sync_to_cluster: Whether to sync to cluster nodes

        Returns:
            Path to the promoted canonical model, or None if failed
        """
        config_key = f"{board_type}_{num_players}p"
        canonical_path = f"models/canonical_{board_type}_{num_players}p.pth"

        try:
            # Copy model to canonical path
            source = Path(model_path)
            dest = Path(canonical_path)
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing canonical if present
            if dest.exists():
                backup_path = dest.with_suffix(".pth.bak")
                shutil.copy2(dest, backup_path)
                logger.info(f"[AutoPromotion] Backed up existing canonical to {backup_path}")

            shutil.copy2(source, dest)
            logger.info(f"[AutoPromotion] Copied {source} -> {dest}")

            # Sync to cluster if requested
            if sync_to_cluster:
                await self._sync_to_cluster(canonical_path, config_key)

            return str(dest)

        except Exception as e:
            logger.error(f"[AutoPromotion] Promotion execution failed: {e}")
            return None

    async def _sync_to_cluster(self, model_path: str, config_key: str) -> None:
        """Sync promoted model to cluster nodes.

        Args:
            model_path: Path to the promoted model
            config_key: Configuration key
        """
        try:
            from app.coordination.unified_distribution_daemon import (
                get_distribution_daemon,
            )
            daemon = get_distribution_daemon()
            if daemon:
                await daemon.distribute_model(model_path, config_key)
                logger.info(f"[AutoPromotion] Model synced to cluster")
        except ImportError:
            logger.debug("[AutoPromotion] Distribution daemon not available, skipping sync")
        except Exception as e:
            logger.warning(f"[AutoPromotion] Cluster sync failed: {e}")

    def _emit_promotion_event(self, event_type: str, result: PromotionResult) -> None:
        """Emit promotion event for feedback loops.

        Args:
            event_type: Event type (MODEL_AUTO_PROMOTED or MODEL_PROMOTION_REJECTED)
            result: Promotion result
        """
        router = self._ensure_event_router()
        if router is None:
            return

        try:
            event_data = {
                "model_path": result.model_path,
                "board_type": result.board_type,
                "num_players": result.num_players,
                "config_key": f"{result.board_type}_{result.num_players}p",
                "approved": result.decision.approved,
                "reason": result.decision.reason,
                "criterion_met": result.decision.criterion_met.value if result.decision.criterion_met else None,
                "timestamp": result.timestamp.isoformat(),
            }

            if result.eval_results:
                event_data.update({
                    "estimated_elo": result.eval_results.estimated_elo,
                    "win_rate_vs_random": result.eval_results.win_rate_vs_random,
                    "win_rate_vs_heuristic": result.eval_results.win_rate_vs_heuristic,
                })

            if result.promoted_path:
                event_data["promoted_path"] = result.promoted_path

            router.publish_sync(event_type, event_data, source="auto_promotion")
            logger.debug(f"[AutoPromotion] Emitted {event_type} event")

        except Exception as e:
            logger.debug(f"[AutoPromotion] Failed to emit event: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


_engine_instance: AutoPromotionEngine | None = None


def get_auto_promotion_engine() -> AutoPromotionEngine:
    """Get or create the singleton AutoPromotionEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AutoPromotionEngine()
    return _engine_instance


def reset_auto_promotion_engine() -> None:
    """Reset the singleton instance (for testing)."""
    global _engine_instance
    _engine_instance = None


async def evaluate_and_promote(
    model_path: str,
    board_type: str,
    num_players: int,
    games: int = 30,
    sync_to_cluster: bool = True,
) -> PromotionResult:
    """Convenience function to evaluate and promote a trained model.

    This is the recommended entry point for automated promotion.

    Args:
        model_path: Path to the trained model checkpoint
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        games: Number of games per opponent for evaluation
        sync_to_cluster: Whether to sync promoted model to cluster

    Returns:
        PromotionResult with decision and details
    """
    engine = get_auto_promotion_engine()
    return await engine.evaluate_and_promote(
        model_path=model_path,
        board_type=board_type,
        num_players=num_players,
        games=games,
        sync_to_cluster=sync_to_cluster,
    )
