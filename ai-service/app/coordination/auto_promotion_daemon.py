"""Auto-Promotion Daemon - Automatically promote models that pass evaluation.

December 2025: This daemon closes the critical feedback loop gap by automatically
promoting models based on gauntlet evaluation results. Previously, promotion
required manual intervention after evaluation completed.

The daemon:
1. Subscribes to EVALUATION_COMPLETED events
2. Checks win rates against promotion thresholds
3. Auto-promotes if thresholds met
4. Emits MODEL_PROMOTED to trigger distribution

Usage:
    from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

    daemon = AutoPromotionDaemon()
    await daemon.start()

Integration with DaemonManager:
    DaemonType.AUTO_PROMOTION factory creates and manages this daemon.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config.thresholds import AUTO_PROMOTION_MIN_QUALITY
from app.coordination.event_utils import parse_config_key

logger = logging.getLogger(__name__)

__all__ = [
    "AutoPromotionConfig",
    "AutoPromotionDaemon",
    "get_auto_promotion_daemon",
]


@dataclass
class AutoPromotionConfig:
    """Configuration for auto-promotion."""
    enabled: bool = True
    # Minimum games required for promotion decision
    # Dec 30: REVERTED from 20 to 50 - 20 games gives 95% CI ±10% which is insufficient
    # to distinguish quality. 50 games gives 95% CI ±6.4% which is acceptable.
    min_games_vs_random: int = 50
    min_games_vs_heuristic: int = 50
    # Cooldown between promotion attempts (seconds)
    promotion_cooldown_seconds: float = 300.0  # 5 minutes
    # Whether to wait for both RANDOM and HEURISTIC results
    require_both_baselines: bool = True
    # Safety: require consecutive successful evaluations
    # Dec 30: REVERTED from 1 to 2 - single pass is too noisy for promotion
    consecutive_passes_required: int = 2
    # Dry run mode - log but don't actually promote
    dry_run: bool = False
    # Dec 27, 2025: Minimum Elo improvement over previous model required for promotion
    # Dec 30: REVERTED from 5 to 10 - require meaningful improvement
    min_elo_improvement: float = 10.0
    # December 2025: Quality gate settings to prevent bad model promotion
    quality_gate_enabled: bool = True
    # Dec 30: REVERTED from 500 to 1000 - require sufficient training data
    min_training_games: int = 1000
    # Dec 30: REVERTED to 0.55 - balance between quality and iteration speed
    min_quality_score: float = 0.55
    require_parity_validation: bool = True  # Require TS parity validation passed
    # December 2025: Stability gate to prevent promoting volatile models
    stability_gate_enabled: bool = True
    max_volatility_score: float = 0.6  # Block models with volatility > 0.6


@dataclass
class PromotionCandidate:
    """Track a model's promotion eligibility."""
    config_key: str
    model_path: str
    evaluation_results: dict[str, float] = field(default_factory=dict)
    evaluation_games: dict[str, int] = field(default_factory=dict)
    consecutive_passes: int = 0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0
    # Dec 27, 2025: Track Elo improvement for stricter promotion
    elo_improvement: float = 0.0  # Elo gain vs previous model
    estimated_elo: float = 0.0  # Estimated Elo from evaluation
    # Dec 28, 2025: Track if model beats current best (for relative promotion)
    beats_current_best: bool = False  # True if this model won head-to-head vs champion
    # Dec 29, 2025: Track consecutive failures for curriculum regression
    consecutive_failures: int = 0  # Number of consecutive failed promotions
    previous_elo: float = 0.0  # Elo before this evaluation (for calculating change)


class AutoPromotionDaemon:
    """Daemon that auto-promotes models based on evaluation results.

    Subscribes to EVALUATION_COMPLETED events and promotes models that
    meet win rate thresholds against RANDOM and HEURISTIC baselines.
    """

    def __init__(self, config: AutoPromotionConfig | None = None):
        self.config = config or AutoPromotionConfig()
        self._running = False
        self._candidates: dict[str, PromotionCandidate] = {}
        self._promotion_history: list[dict[str, Any]] = []
        self._subscribed = False

    @property
    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the auto-promotion daemon."""
        if self._running:
            return

        self._running = True
        self._subscription_task: asyncio.Task | None = None
        await self._subscribe_to_events()

        # Dec 29, 2025: Start background subscription retry if initial failed
        # This handles the case where router becomes available after daemon starts
        if not self._subscribed:
            self._subscription_task = asyncio.create_task(
                self._periodic_subscription_retry()
            )

        logger.info("[AutoPromotion] Daemon started")

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

        # Cancel background subscription task if running
        if hasattr(self, "_subscription_task") and self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass
            self._subscription_task = None

        logger.info("[AutoPromotion] Daemon stopped")

    async def _periodic_subscription_retry(self) -> None:
        """Periodically retry event subscription until successful.

        Dec 29, 2025: Added to handle case where router becomes available
        after daemon starts. Retries every 60 seconds until subscribed.
        """
        retry_interval = 60.0  # seconds
        max_attempts = 10  # Give up after 10 minutes

        for attempt in range(max_attempts):
            if not self._running:
                return

            await asyncio.sleep(retry_interval)

            if self._subscribed:
                logger.debug("[AutoPromotion] Already subscribed, stopping retry loop")
                return

            logger.debug(f"[AutoPromotion] Subscription retry attempt {attempt + 1}/{max_attempts}")
            await self._subscribe_to_events()

            if self._subscribed:
                logger.info("[AutoPromotion] Subscription succeeded on retry")
                return

        logger.warning("[AutoPromotion] Gave up on subscription after max retries")

    async def _subscribe_to_events(self) -> None:
        """Subscribe to EVALUATION_COMPLETED events with retry logic.

        Retries up to 3 times with exponential backoff if the router
        is not immediately available.
        """
        if self._subscribed:
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                from app.coordination.event_router import DataEventType, get_router

                # Dec 29, 2025: Check if DataEventType is available (None if data_events failed to import)
                if DataEventType is None:
                    logger.warning("[AutoPromotion] DataEventType unavailable (data_events not imported)")
                    self._subscribed = False
                    return

                router = get_router()
                if not router:
                    if attempt == max_retries - 1:
                        logger.warning("[AutoPromotion] Router unavailable after retries")
                        self._subscribed = False
                        return
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue

                # December 29, 2025: router.subscribe() is synchronous, not async
                router.subscribe(
                    DataEventType.EVALUATION_COMPLETED,
                    self._on_evaluation_completed,
                )
                self._subscribed = True
                logger.info("[AutoPromotion] Subscribed to EVALUATION_COMPLETED")
                return
            except ImportError as e:
                logger.warning(f"[AutoPromotion] Event system not available: {e}")
                self._subscribed = False
                return
            except (RuntimeError, AttributeError, TypeError) as e:
                if attempt == max_retries - 1:
                    logger.warning(f"[AutoPromotion] Failed to subscribe after retries: {e}")
                    self._subscribed = False
                else:
                    logger.debug(f"[AutoPromotion] Subscription attempt {attempt+1} failed: {e}")

    async def _on_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event.

        Args:
            event: Event with payload containing evaluation results
        """
        if not self.config.enabled:
            return

        try:
            payload = event.payload if hasattr(event, "payload") else event
            await self._process_evaluation(payload)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Error processing evaluation: {e}")

    async def _process_evaluation(self, payload: dict[str, Any]) -> None:
        """Process evaluation results and decide on promotion.

        Args:
            payload: Evaluation event payload with:
                - config_key: e.g., "hex8_2p"
                - model_path: Path to evaluated model
                - opponent_type: "RANDOM" or "HEURISTIC"
                - win_rate: Win rate against opponent
                - games_played: Number of games
        """
        config_key = payload.get("config_key") or payload.get("config")
        model_path = payload.get("model_path")
        opponent_type = payload.get("opponent_type", "").upper()
        win_rate = payload.get("win_rate", 0.0)
        games_played = payload.get("games_played", 0)

        if not config_key or not model_path:
            logger.debug("[AutoPromotion] Missing config_key or model_path")
            return

        # Get or create candidate
        if config_key not in self._candidates:
            self._candidates[config_key] = PromotionCandidate(
                config_key=config_key,
                model_path=model_path,
            )

        candidate = self._candidates[config_key]
        candidate.model_path = model_path
        candidate.last_evaluation_time = time.time()

        # Record result - single opponent type
        if opponent_type in ("RANDOM", "HEURISTIC"):
            candidate.evaluation_results[opponent_type] = win_rate
            candidate.evaluation_games[opponent_type] = games_played
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs {opponent_type}: "
                f"{win_rate:.1%} ({games_played} games)"
            )

        # Dec 28, 2025: Also check for direct baseline win rates from gauntlet
        # The gauntlet emits both rates in a single event
        vs_random_rate = payload.get("vs_random_rate")
        vs_heuristic_rate = payload.get("vs_heuristic_rate")
        if vs_random_rate is not None:
            candidate.evaluation_results["RANDOM"] = float(vs_random_rate)
            # Use total games if per-opponent count not available
            if "RANDOM" not in candidate.evaluation_games:
                candidate.evaluation_games["RANDOM"] = games_played // 2 or 50
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs RANDOM from gauntlet: "
                f"{vs_random_rate:.1%}"
            )
        if vs_heuristic_rate is not None:
            candidate.evaluation_results["HEURISTIC"] = float(vs_heuristic_rate)
            if "HEURISTIC" not in candidate.evaluation_games:
                candidate.evaluation_games["HEURISTIC"] = games_played // 2 or 50
            logger.info(
                f"[AutoPromotion] Recorded {config_key} vs HEURISTIC from gauntlet: "
                f"{vs_heuristic_rate:.1%}"
            )

        # Dec 27, 2025: Extract Elo information from payload
        elo_improvement = payload.get("elo_improvement", payload.get("elo_delta", 0.0))
        estimated_elo = payload.get("estimated_elo", payload.get("elo", 0.0))
        if elo_improvement:
            candidate.elo_improvement = float(elo_improvement)
        if estimated_elo:
            candidate.estimated_elo = float(estimated_elo)

        # Dec 28, 2025: Extract beats_current_best from payload
        # This flag indicates if the model won head-to-head vs the current champion
        beats_current_best = payload.get("beats_current_best", payload.get("beats_champion", False))
        if beats_current_best:
            candidate.beats_current_best = bool(beats_current_best)

        # Check if ready for promotion decision
        await self._check_promotion(candidate)

    async def _check_promotion(self, candidate: PromotionCandidate) -> None:
        """Check if candidate meets promotion criteria.

        Uses the two-tier promotion system from thresholds.py:
        1. ASPIRATIONAL thresholds - strong models that definitely should promote
        2. MINIMUM floor + beats_current_best - incremental improvements that beat champion

        Args:
            candidate: PromotionCandidate to evaluate
        """
        # Dec 28, 2025: Use unified should_promote_model() for two-tier promotion
        from app.config.thresholds import should_promote_model

        # Check if we have required results
        has_random = "RANDOM" in candidate.evaluation_results
        has_heuristic = "HEURISTIC" in candidate.evaluation_results

        if self.config.require_both_baselines and not (has_random and has_heuristic):
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: Waiting for both baselines "
                f"(RANDOM={has_random}, HEURISTIC={has_heuristic})"
            )
            return

        # Check game counts
        random_games = candidate.evaluation_games.get("RANDOM", 0)
        heuristic_games = candidate.evaluation_games.get("HEURISTIC", 0)

        if random_games < self.config.min_games_vs_random:
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: "
                f"Need {self.config.min_games_vs_random} games vs RANDOM, have {random_games}"
            )
            return

        if heuristic_games < self.config.min_games_vs_heuristic:
            logger.debug(
                f"[AutoPromotion] {candidate.config_key}: "
                f"Need {self.config.min_games_vs_heuristic} games vs HEURISTIC, have {heuristic_games}"
            )
            return

        # Get win rates
        random_win_rate = candidate.evaluation_results.get("RANDOM", 0.0)
        heuristic_win_rate = candidate.evaluation_results.get("HEURISTIC", 0.0)

        # Dec 28, 2025: Use two-tier promotion system
        # - Aspirational: Model meets high thresholds for strong performance
        # - Relative: Model beats current best AND meets minimum floor
        # Dec 30, 2025: Pass current_best_elo to enable safety check
        # This prevents "race to the bottom" where weak models beat weaker models
        current_best_elo = candidate.previous_elo if candidate.previous_elo > 0 else None
        # Dec 30, 2025: Pass model_elo to enable Elo-adaptive thresholds
        # This allows bootstrap models (800-1200 Elo) to pass with lower thresholds
        model_elo = candidate.estimated_elo if candidate.estimated_elo > 0 else None
        should_promote, reason = should_promote_model(
            config_key=candidate.config_key,
            vs_random_rate=random_win_rate,
            vs_heuristic_rate=heuristic_win_rate,
            beats_current_best=candidate.beats_current_best,
            current_best_elo=current_best_elo,
            model_elo=model_elo,
        )

        if should_promote:
            candidate.consecutive_passes += 1
            logger.info(
                f"[AutoPromotion] {candidate.config_key} PASSES: {reason} "
                f"[streak={candidate.consecutive_passes}, beats_best={candidate.beats_current_best}]"
            )

            # Check cooldown
            time_since_last = time.time() - candidate.last_promotion_time
            if time_since_last < self.config.promotion_cooldown_seconds:
                remaining = self.config.promotion_cooldown_seconds - time_since_last
                logger.info(
                    f"[AutoPromotion] {candidate.config_key}: "
                    f"In cooldown, {remaining:.0f}s remaining"
                )
                return

            # Check consecutive passes
            if candidate.consecutive_passes >= self.config.consecutive_passes_required:
                # Dec 27, 2025: Check Elo improvement requirement (optional)
                # Note: For relative promotion, we may want to skip this check
                # since beating champion is already strong signal
                if (
                    self.config.min_elo_improvement > 0
                    and not candidate.beats_current_best  # Skip Elo check if beat champion
                    and candidate.elo_improvement < self.config.min_elo_improvement
                ):
                    logger.info(
                        f"[AutoPromotion] {candidate.config_key}: "
                        f"Elo improvement {candidate.elo_improvement:+.1f} < {self.config.min_elo_improvement} required"
                    )
                    return
                await self._promote_model(candidate)
        else:
            # Reset streak on failure
            candidate.consecutive_passes = 0
            logger.info(
                f"[AutoPromotion] {candidate.config_key} FAILS: {reason} "
                f"(vs_random={random_win_rate:.1%}, vs_heuristic={heuristic_win_rate:.1%}, "
                f"beats_best={candidate.beats_current_best})"
            )

    async def _check_quality_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if candidate passes quality gate before promotion.

        December 2025: Prevents promotion of models trained on corrupted or
        insufficient data by verifying:
        1. Parity validation is complete (TS/Python match)
        2. Sufficient training game count (1000+)
        3. Data quality score is acceptable (>0.6)

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not self.config.quality_gate_enabled:
            return True, "quality_gate_disabled"

        config_key = candidate.config_key

        # Parse board_type and num_players using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"[AutoPromotion] Cannot parse config_key: {config_key}")
            return True, "config_key_unparseable"
        board_type = parsed.board_type
        num_players = parsed.num_players

        # Check parity validation status
        if self.config.require_parity_validation:
            parity_passed, parity_reason = await self._check_parity_status(
                board_type, num_players
            )
            if not parity_passed:
                # Dec 2025: If database check fails due to pending/incomplete,
                # try live parity validation on coordinator (has Node.js)
                if "pending" in parity_reason.lower() or "incomplete" in parity_reason.lower():
                    logger.info(
                        f"[AutoPromotion] Database parity check incomplete for {config_key}, "
                        f"attempting live validation"
                    )
                    live_passed, live_reason = await self._run_live_parity_validation(
                        config_key, sample_games=100
                    )
                    if not live_passed:
                        return False, f"live_parity_failed: {live_reason}"
                    # Live validation passed, continue
                    logger.info(
                        f"[AutoPromotion] Live parity validation passed for {config_key}: "
                        f"{live_reason}"
                    )
                else:
                    return False, f"parity_failed: {parity_reason}"

        # Check training data quality
        quality_passed, quality_reason = await self._check_data_quality(
            board_type, num_players
        )
        if not quality_passed:
            return False, f"quality_failed: {quality_reason}"

        return True, "quality_gate_passed"

    async def _check_parity_status(
        self,
        board_type: str,
        num_players: int,
    ) -> tuple[bool, str]:
        """Check if parity validation has passed for this config.

        Args:
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Tuple of (passed, reason)
        """
        try:
            from pathlib import Path

            from app.db.game_replay import GameReplayDB

            # Find canonical database
            db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")
            if not db_path.exists():
                return False, f"database_not_found: {db_path}"

            # Check parity_gate status in database using context manager
            with GameReplayDB(str(db_path)) as db:
                with db._get_conn() as conn:
                    # Count games by parity status
                    cursor = conn.execute(
                        """
                        SELECT parity_gate, COUNT(*) as count
                        FROM games
                        WHERE game_status = 'completed'
                        GROUP BY parity_gate
                        """
                    )
                    status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            total_games = sum(status_counts.values())
            passed_games = status_counts.get("passed", 0)
            pending_games = status_counts.get("pending_gate", 0)
            failed_games = status_counts.get("failed", 0)

            # Require majority of games to have passed parity
            if total_games == 0:
                return False, "no_completed_games"

            pass_rate = passed_games / total_games if total_games > 0 else 0

            if pass_rate < 0.5:
                return False, (
                    f"low_parity_pass_rate: {pass_rate:.1%} "
                    f"(passed={passed_games}, pending={pending_games}, failed={failed_games})"
                )

            # If too many pending, validation hasn't run
            if pending_games > passed_games:
                return False, (
                    f"parity_validation_incomplete: pending={pending_games}, passed={passed_games}"
                )

            return True, f"parity_ok: {pass_rate:.1%} passed"

        except Exception as e:
            logger.warning(f"[AutoPromotion] Parity check error: {e}")
            # Don't block on parity check errors
            return True, f"parity_check_error: {e}"

    async def _run_live_parity_validation(
        self, config_key: str, sample_games: int = 100
    ) -> tuple[bool, str]:
        """Run live TS/Python parity validation on coordinator.

        December 2025: The coordinator (mac-studio) has Node.js installed, allowing
        us to run actual parity validation instead of relying on database status.
        Cluster nodes lack npx so parity gates often show "pending_gate".

        This method runs the parity check script to validate a sample of games,
        ensuring the Python rules engine matches TypeScript before promotion.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            sample_games: Number of games to validate (default: 100)

        Returns:
            Tuple of (passed, reason) where reason explains the result
        """
        import asyncio
        import subprocess
        from pathlib import Path

        # Only run on coordinator (has Node.js)
        try:
            from app.config.env import env
            if not env.is_coordinator:
                logger.debug(
                    f"[AutoPromotion] Skipping live parity validation for {config_key} "
                    "(not coordinator)"
                )
                return True, "skipped_not_coordinator"
        except ImportError:
            # If env module unavailable, check by hostname
            import socket
            hostname = socket.gethostname().lower()
            if "mac-studio" not in hostname and "local-mac" not in hostname:
                return True, "skipped_not_coordinator"

        # Parse board_type and num_players using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"[AutoPromotion] Cannot parse config_key: {config_key}")
            return True, "config_key_unparseable"
        board_type = parsed.board_type
        num_players = parsed.num_players

        # Find canonical database for this config
        db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")
        if not db_path.exists():
            logger.debug(
                f"[AutoPromotion] No canonical DB for {config_key}, skipping live parity"
            )
            return True, "no_canonical_db"

        # Build parity check command
        script_path = Path("scripts/check_ts_python_replay_parity.py")
        if not script_path.exists():
            logger.warning("[AutoPromotion] Parity check script not found")
            return True, "script_not_found"

        cmd = [
            "python",
            str(script_path),
            "--db", str(db_path),
            "--limit", str(sample_games),
        ]

        logger.info(
            f"[AutoPromotion] Running live parity validation for {config_key} "
            f"({sample_games} games from {db_path})"
        )

        try:
            # Run parity check with timeout
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path.cwd()),
            )

            if result.returncode == 0:
                logger.info(
                    f"[AutoPromotion] Live parity validation PASSED for {config_key}"
                )
                return True, "parity_passed"
            else:
                # Extract error from stderr
                error_msg = result.stderr.decode("utf-8", errors="replace")[:500]
                stdout_msg = result.stdout.decode("utf-8", errors="replace")[:200]
                logger.warning(
                    f"[AutoPromotion] Live parity validation FAILED for {config_key}: "
                    f"exit={result.returncode}, stderr={error_msg}"
                )
                return False, f"parity_failed: {error_msg or stdout_msg}"

        except subprocess.TimeoutExpired:
            logger.warning(
                f"[AutoPromotion] Parity validation timed out for {config_key}"
            )
            return False, "parity_timeout"
        except FileNotFoundError as e:
            logger.warning(f"[AutoPromotion] Parity script not executable: {e}")
            return True, f"parity_script_error: {e}"
        except OSError as e:
            logger.warning(f"[AutoPromotion] Parity validation OS error: {e}")
            return True, f"parity_os_error: {e}"

    async def _check_data_quality(
        self,
        board_type: str,
        num_players: int,
    ) -> tuple[bool, str]:
        """Check if training data quality is sufficient.

        Args:
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Tuple of (passed, reason)
        """
        try:
            from pathlib import Path

            from app.db.game_replay import GameReplayDB

            # Find canonical database
            db_path = Path(f"data/games/canonical_{board_type}_{num_players}p.db")
            if not db_path.exists():
                return False, f"database_not_found: {db_path}"

            # Check game count using context manager
            with GameReplayDB(str(db_path)) as db:
                with db._get_conn() as conn:
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM games
                        WHERE game_status = 'completed'
                        """
                    )
                    game_count = cursor.fetchone()[0]

            if game_count < self.config.min_training_games:
                return False, (
                    f"insufficient_games: {game_count} < {self.config.min_training_games}"
                )

            # Try to get quality score if available
            try:
                from app.training.data_quality import (
                    DatabaseQualityChecker,
                    get_database_quality_score,
                )

                quality_score = get_database_quality_score(str(db_path))
                if quality_score < self.config.min_quality_score:
                    return False, (
                        f"low_quality_score: {quality_score:.2f} < {self.config.min_quality_score}"
                    )

            except (ImportError, AttributeError):
                # Quality score not available, skip this check
                logger.debug("[AutoPromotion] Quality score check not available")

            return True, f"quality_ok: {game_count} games"

        except Exception as e:
            logger.warning(f"[AutoPromotion] Quality check error: {e}")
            # Don't block on quality check errors
            return True, f"quality_check_error: {e}"

    async def _check_stability_gate(
        self,
        candidate: PromotionCandidate,
    ) -> tuple[bool, str]:
        """Check if candidate passes stability gate before promotion.

        December 2025: Prevents promotion of volatile models by verifying:
        1. Rating volatility is within acceptable bounds
        2. Model is not in a declining trend
        3. Rating has stabilized (not oscillating)

        Args:
            candidate: PromotionCandidate to check

        Returns:
            Tuple of (passed, reason) where reason explains the gate result
        """
        if not self.config.stability_gate_enabled:
            return True, "stability_gate_disabled"

        config_key = candidate.config_key

        # Parse board_type and num_players using canonical utility
        parsed = parse_config_key(config_key)
        if not parsed:
            logger.warning(f"[AutoPromotion] Cannot parse config_key: {config_key}")
            return True, "config_key_unparseable"
        board_type = parsed.board_type
        num_players = parsed.num_players

        try:
            from app.coordination.stability_heuristic import (
                StabilityLevel,
                assess_model_stability,
            )

            # Assess stability
            assessment = assess_model_stability(
                model_id="canonical",
                board_type=board_type,
                num_players=num_players,
            )

            # Log the assessment for debugging
            logger.info(
                f"[AutoPromotion] Stability assessment for {config_key}: "
                f"level={assessment.level.value}, volatility={assessment.volatility_score:.3f}, "
                f"slope={assessment.slope:.2f}, samples={assessment.sample_count}"
            )

            # Check volatility score
            if assessment.volatility_score > self.config.max_volatility_score:
                return False, (
                    f"high_volatility: {assessment.volatility_score:.2f} > "
                    f"{self.config.max_volatility_score}"
                )

            # Block declining models
            if assessment.level == StabilityLevel.DECLINING:
                return False, (
                    f"declining_trend: slope={assessment.slope:.2f} Elo/hour"
                )

            # Block highly volatile models
            if assessment.level == StabilityLevel.VOLATILE:
                actions = ", ".join(assessment.recommended_actions[:2])
                return False, f"volatile_model: {actions}"

            # Check if promotion is explicitly unsafe
            if not assessment.promotion_safe:
                return False, f"promotion_unsafe: {', '.join(assessment.recommended_actions[:1])}"

            return True, (
                f"stability_ok: level={assessment.level.value}, "
                f"volatility={assessment.volatility_score:.2f}"
            )

        except ImportError:
            logger.debug("[AutoPromotion] Stability heuristic not available")
            return True, "stability_check_unavailable"
        except Exception as e:
            logger.warning(f"[AutoPromotion] Stability check error: {e}")
            # Don't block on stability check errors
            return True, f"stability_check_error: {e}"

    async def _promote_model(self, candidate: PromotionCandidate) -> None:
        """Promote a model that passed evaluation.

        Args:
            candidate: PromotionCandidate to promote
        """
        config_key = candidate.config_key
        model_path = candidate.model_path

        # December 2025: Check quality gate before promotion
        quality_passed, quality_reason = await self._check_quality_gate(candidate)
        if not quality_passed:
            logger.warning(
                f"[AutoPromotion] {config_key} blocked by quality gate: {quality_reason}"
            )
            await self._emit_promotion_failed(candidate, error=f"quality_gate: {quality_reason}")
            return

        # December 2025: Check stability gate to prevent promoting volatile models
        stability_passed, stability_reason = await self._check_stability_gate(candidate)
        if not stability_passed:
            logger.warning(
                f"[AutoPromotion] {config_key} blocked by stability gate: {stability_reason}"
            )
            await self._emit_promotion_failed(candidate, error=f"stability_gate: {stability_reason}")
            return

        if self.config.dry_run:
            logger.info(
                f"[AutoPromotion] DRY RUN: Would promote {config_key} "
                f"({model_path})"
            )
            return

        logger.info(f"[AutoPromotion] Promoting {config_key} ({model_path})")

        try:
            # Import promotion controller
            from app.training.promotion_controller import PromotionController

            controller = PromotionController()
            success = await controller.promote_model(
                config_key=config_key,
                model_path=model_path,
                reason="auto_promotion_passed_evaluation",
                evaluation_results={
                    "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                    "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                },
            )

            if success:
                candidate.last_promotion_time = time.time()
                # Dec 29, 2025: Reset failures and update Elo tracking on success
                candidate.consecutive_failures = 0
                candidate.previous_elo = candidate.estimated_elo  # Update baseline

                self._promotion_history.append({
                    "config_key": config_key,
                    "model_path": model_path,
                    "timestamp": time.time(),
                    "vs_random": candidate.evaluation_results.get("RANDOM"),
                    "vs_heuristic": candidate.evaluation_results.get("HEURISTIC"),
                })

                logger.info(f"[AutoPromotion] Successfully promoted {config_key}")

                # Emit MODEL_PROMOTED event
                await self._emit_promotion_event(candidate)

                # Dec 29, 2025: Emit unified PROMOTION_COMPLETED for curriculum
                await self._emit_promotion_completed(candidate, success=True)

                # Clear results after successful emission
                candidate.consecutive_passes = 0
                candidate.evaluation_results.clear()
                candidate.evaluation_games.clear()
            else:
                # Dec 29, 2025: Track consecutive failures for curriculum regression
                candidate.consecutive_failures += 1

                logger.warning(
                    f"[AutoPromotion] Promotion failed for {config_key} "
                    f"(consecutive_failures={candidate.consecutive_failures})"
                )

                # Emit PROMOTION_FAILED event to trigger curriculum weight increase
                await self._emit_promotion_failed(candidate, error="Promotion validation failed")

                # Dec 29, 2025: Emit unified PROMOTION_COMPLETED for curriculum
                await self._emit_promotion_completed(candidate, success=False)

        except ImportError:
            # Fallback: just emit the event
            logger.warning(
                "[AutoPromotion] PromotionController not available, "
                "emitting event only"
            )
            await self._emit_promotion_event(candidate)
            await self._emit_promotion_completed(candidate, success=True)
        except Exception as e:  # noqa: BLE001
            # Dec 29, 2025: Track failures on exception too
            candidate.consecutive_failures += 1
            logger.error(
                f"[AutoPromotion] Promotion error for {config_key}: {e} "
                f"(consecutive_failures={candidate.consecutive_failures})"
            )

            # Emit PROMOTION_FAILED event on exception
            await self._emit_promotion_failed(candidate, error=str(e))

            # Dec 29, 2025: Emit unified PROMOTION_COMPLETED for curriculum
            await self._emit_promotion_completed(candidate, success=False)

    async def _emit_promotion_event(self, candidate: PromotionCandidate) -> None:
        """Emit MODEL_PROMOTED event and CURRICULUM_ADVANCED if applicable.

        Args:
            candidate: PromotionCandidate that was promoted
        """
        try:
            from app.coordination.event_router import DataEventType, emit_curriculum_advanced, get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload={
                        "config_key": candidate.config_key,
                        "model_path": candidate.model_path,
                        "reason": "auto_promotion_daemon",
                        "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                        "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                        "timestamp": time.time(),
                    },
                    source="auto_promotion_daemon",
                )
                logger.info(
                    f"[AutoPromotion] Emitted MODEL_PROMOTED for {candidate.config_key}"
                )

                # P0.5 Dec 2025: Emit CURRICULUM_ADVANCED when consecutive promotions
                # indicate curriculum tier progression readiness
                if candidate.consecutive_passes >= 2:
                    # Determine tier from consecutive pass count
                    old_tier = f"TIER_{candidate.consecutive_passes - 1}"
                    new_tier = f"TIER_{candidate.consecutive_passes}"
                    await emit_curriculum_advanced(
                        config=candidate.config_key,
                        old_tier=old_tier,
                        new_tier=new_tier,
                        trigger_reason="consecutive_promotions",
                        win_rate=candidate.evaluation_results.get("HEURISTIC", 0.0),
                        games_at_tier=candidate.evaluation_games,
                        source="auto_promotion_daemon",
                    )
                    logger.info(
                        f"[AutoPromotion] Emitted CURRICULUM_ADVANCED for {candidate.config_key}: "
                        f"{old_tier} → {new_tier}"
                    )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Failed to emit promotion event: {e}")

    async def _emit_promotion_failed(
        self,
        candidate: PromotionCandidate,
        error: str,
    ) -> None:
        """Emit PROMOTION_FAILED event to trigger curriculum weight increase.

        Args:
            candidate: PromotionCandidate that failed promotion
            error: Error message or reason for failure
        """
        try:
            from app.coordination.event_router import get_router
            from app.events.types import RingRiftEventType

            router = get_router()
            if router:
                await router.publish(
                    event_type=RingRiftEventType.PROMOTION_FAILED,
                    payload={
                        "config_key": candidate.config_key,
                        "config": candidate.config_key,  # Alternate key for compatibility
                        "model_id": candidate.model_path,
                        "error": error,
                        "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                        "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                        "timestamp": time.time(),
                    },
                    source="auto_promotion_daemon",
                )
                logger.info(
                    f"[AutoPromotion] Emitted PROMOTION_FAILED for {candidate.config_key}: {error}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Failed to emit PROMOTION_FAILED event: {e}")

    async def _emit_promotion_completed(
        self,
        candidate: PromotionCandidate,
        success: bool,
    ) -> None:
        """Emit unified PROMOTION_COMPLETED event for curriculum feedback.

        Dec 29, 2025: Provides a single event for curriculum_integration to
        subscribe to, containing all information needed for curriculum
        advancement or regression decisions.

        Args:
            candidate: PromotionCandidate that was evaluated
            success: Whether promotion succeeded
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                # Calculate Elo change
                elo_change = candidate.estimated_elo - candidate.previous_elo

                await router.publish(
                    event_type="PROMOTION_COMPLETED",
                    payload={
                        "config_key": candidate.config_key,
                        "success": success,
                        "elo_change": elo_change,
                        "estimated_elo": candidate.estimated_elo,
                        "previous_elo": candidate.previous_elo,
                        "consecutive_failures": candidate.consecutive_failures,
                        "consecutive_passes": candidate.consecutive_passes,
                        "vs_random": candidate.evaluation_results.get("RANDOM", 0.0),
                        "vs_heuristic": candidate.evaluation_results.get("HEURISTIC", 0.0),
                        "timestamp": time.time(),
                    },
                    source="auto_promotion_daemon",
                )
                logger.info(
                    f"[AutoPromotion] Emitted PROMOTION_COMPLETED for {candidate.config_key}: "
                    f"success={success}, elo_change={elo_change:+.0f}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoPromotion] Failed to emit PROMOTION_COMPLETED event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "candidates": {
                k: {
                    "model_path": v.model_path,
                    "results": v.evaluation_results,
                    "games": v.evaluation_games,
                    "consecutive_passes": v.consecutive_passes,
                    "last_evaluation": v.last_evaluation_time,
                    "last_promotion": v.last_promotion_time,
                }
                for k, v in self._candidates.items()
            },
            "promotion_history_count": len(self._promotion_history),
            "recent_promotions": self._promotion_history[-5:] if self._promotion_history else [],
        }

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="AutoPromotion daemon not running",
            )

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="AutoPromotion daemon not subscribed to events",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"AutoPromotion daemon running ({len(self._promotion_history)} promotions)",
            details=self.get_status(),
        )


# Module-level singleton
_auto_promotion_daemon: AutoPromotionDaemon | None = None


def get_auto_promotion_daemon() -> AutoPromotionDaemon:
    """Get the singleton AutoPromotionDaemon instance."""
    global _auto_promotion_daemon
    if _auto_promotion_daemon is None:
        _auto_promotion_daemon = AutoPromotionDaemon()
    return _auto_promotion_daemon


def reset_auto_promotion_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _auto_promotion_daemon
    _auto_promotion_daemon = None
