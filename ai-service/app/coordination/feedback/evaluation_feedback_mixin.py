"""Evaluation feedback handling for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~320 LOC)

This mixin provides evaluation-related feedback logic that:
- Handles evaluation completion and failure events
- Triggers multi-harness gauntlet evaluations after training
- Implements retry logic for failed evaluations
- Triggers baseline gauntlet re-evaluation after regressions

The evaluation feedback loop closes the training -> evaluation -> promotion cycle:
TRAINING_COMPLETED -> _trigger_evaluation -> gauntlet -> EVALUATION_COMPLETED
                                                       -> _consider_promotion

Usage:
    class FeedbackLoopController(EvaluationFeedbackMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.coordination.event_utils import parse_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)


def _safe_create_task(coro, context: str = "") -> asyncio.Task | None:
    """Create a task with basic error handling.

    Note: This is a local helper. The main controller has a more sophisticated
    version with error tracking. This is used for mixin independence.
    """
    try:
        task = asyncio.create_task(coro)
        task.add_done_callback(
            lambda t: logger.debug(f"[EvaluationFeedback] Task {context} done")
            if not t.cancelled() and t.exception() is None
            else logger.warning(f"[EvaluationFeedback] Task {context} failed: {t.exception()}")
            if t.exception() else None
        )
        return task
    except RuntimeError as e:
        logger.debug(f"[EvaluationFeedback] Could not create task for {context}: {e}")
        return None


class EvaluationFeedbackMixin:
    """Mixin for evaluation feedback handling in FeedbackLoopController.

    Requires the host class to implement:
    - _get_or_create_state(config_key: str) -> FeedbackState
    - _consider_promotion(config_key, model_path, win_rate, elo_delta) (for promotion decisions)

    Provides:
    - _trigger_evaluation(config_key, model_path) - Trigger multi-harness gauntlet
    - _run_single_harness_gauntlet(config_key, model_path, board_type, num_players)
    - _trigger_gauntlet_all_baselines(config_key) - Post-regression reassessment
    - _retry_evaluation(config_key, model_path, attempt) - Retry with modified params
    """

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create state for a config. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _get_or_create_state")

    def _consider_promotion(
        self, config_key: str, model_path: str, win_rate: float, elo_delta: float
    ) -> None:
        """Consider model promotion. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _consider_promotion")

    def _trigger_evaluation(self, config_key: str, model_path: str) -> None:
        """Trigger multi-harness gauntlet evaluation automatically after training.

        December 2025: Wires TRAINING_COMPLETED -> auto-gauntlet evaluation.
        This closes the training feedback loop by automatically evaluating
        newly trained models against baselines under ALL compatible harnesses.

        The gauntlet results determine whether the model should be promoted
        to production or if more training is needed. Multi-harness evaluation
        enables finding the best (model, harness) combination for deployment.

        January 28, 2026: On coordinator nodes (gauntlet_enabled=false), dispatches
        gauntlet to work queue for cluster execution instead of running locally.
        """
        logger.info(f"[EvaluationFeedback] Triggering multi-harness evaluation for {config_key}")

        try:
            # Parse config_key using canonical utility
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.warning(f"[EvaluationFeedback] Invalid config_key format: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            # January 28, 2026: Check gauntlet_enabled FIRST
            # On coordinator nodes (gauntlet_enabled=false), dispatch to cluster work queue
            # instead of running heavy gauntlet workloads locally
            from app.config.env import env
            if not env.gauntlet_enabled:
                logger.info(
                    f"[EvaluationFeedback] Coordinator mode (gauntlet_enabled=false): "
                    f"dispatching gauntlet to cluster for {config_key}"
                )
                _safe_create_task(
                    self._dispatch_gauntlet_to_work_queue(config_key, model_path, board_type, num_players),
                    f"dispatch_gauntlet({config_key})"
                )
                return

            # GPU node - run gauntlet locally
            # Launch multi-harness gauntlet evaluation asynchronously
            async def run_multi_harness_gauntlet():
                """Run multi-harness gauntlet evaluation for the trained model.

                December 2025: Evaluates model under ALL compatible harnesses
                (e.g., policy_only, mcts, gumbel_mcts, descent for NN models).
                Registers all harness results in Elo system and uses the best
                harness for promotion decisions.

                Falls back to single-harness evaluation if multi-harness unavailable.
                """
                try:
                    from app.training.multi_harness_gauntlet import (
                        MultiHarnessGauntlet,
                        register_multi_harness_results,
                    )

                    # January 3, 2026 (Session 8): Wait for model distribution before evaluation
                    # This prevents wasting 45-60s evaluating with stale baseline models
                    # on cluster nodes that haven't received the new model yet.
                    try:
                        from app.coordination.unified_distribution_daemon import (
                            wait_for_model_availability,
                        )
                        from app.config.coordination_defaults import DistributionDefaults

                        # Wait for model to be distributed to at least MIN_NODES_FOR_PROMOTION nodes
                        # with a reasonable timeout (default 180s for distribution + buffer)
                        distribution_timeout = getattr(
                            DistributionDefaults, "DISTRIBUTION_TIMEOUT_SECONDS", 180.0
                        )
                        success, node_count = await wait_for_model_availability(
                            model_path=model_path,
                            min_nodes=getattr(DistributionDefaults, "MIN_NODES_FOR_PROMOTION", 3),
                            timeout=distribution_timeout,
                        )

                        if success:
                            logger.info(
                                f"[EvaluationFeedback] Model {config_key} distributed to "
                                f"{node_count} nodes, proceeding with evaluation"
                            )
                        else:
                            logger.warning(
                                f"[EvaluationFeedback] Model {config_key} only on {node_count} nodes "
                                f"after {distribution_timeout}s, proceeding anyway"
                            )
                    except ImportError:
                        logger.debug(
                            "[EvaluationFeedback] Distribution verification not available, "
                            "proceeding with evaluation"
                        )

                    gauntlet = MultiHarnessGauntlet(default_games_per_baseline=30)
                    result = await gauntlet.evaluate_model(
                        model_path=model_path,
                        board_type=board_type,
                        num_players=num_players,
                    )

                    # Register all harness results in Elo system
                    participant_ids = register_multi_harness_results(result)
                    logger.info(
                        f"[EvaluationFeedback] Multi-harness evaluation complete for {config_key}: "
                        f"best={result.best_harness} Elo={result.best_elo:.0f}, "
                        f"harnesses={list(result.harness_results.keys())}"
                    )

                    # Use best harness result for promotion decision
                    if result.best_elo > 0 and result.harness_results:
                        best_rating = result.harness_results.get(result.best_harness)
                        if best_rating:
                            win_rate = getattr(best_rating, "win_rate", 0.0)
                            elo_delta = result.best_elo - 1000  # Delta from baseline
                            if win_rate >= 0.55:  # Minimum threshold for promotion consideration
                                self._consider_promotion(
                                    config_key,
                                    model_path,
                                    win_rate,
                                    elo_delta,
                                )

                except ImportError as e:
                    # Fall back to single-harness evaluation if multi-harness unavailable
                    logger.debug(
                        f"[EvaluationFeedback] MultiHarnessGauntlet not available, "
                        f"falling back to single-harness: {e}"
                    )
                    await self._run_single_harness_gauntlet(
                        config_key, model_path, board_type, num_players
                    )
                except Exception as e:
                    logger.error(
                        f"[EvaluationFeedback] Multi-harness evaluation failed for {config_key}: {e}"
                    )

            _safe_create_task(run_multi_harness_gauntlet(), f"run_multi_harness_gauntlet({config_key})")

        except ImportError as e:
            logger.debug(f"[EvaluationFeedback] trigger_evaluation not available: {e}")

    async def _run_single_harness_gauntlet(
        self, config_key: str, model_path: str, board_type: str, num_players: int
    ) -> None:
        """Run single-harness gauntlet as fallback when MultiHarnessGauntlet unavailable.

        This is the legacy evaluation path using pipeline_actions.trigger_evaluation.
        Used when the multi-harness gauntlet is not available (e.g., missing dependencies).

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            model_path: Path to model checkpoint
            board_type: Board type string
            num_players: Number of players
        """
        try:
            from app.coordination.pipeline_actions import trigger_evaluation

            result = await trigger_evaluation(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                num_games=50,  # Standard gauntlet size
            )
            if result.success:
                logger.info(
                    f"[EvaluationFeedback] Single-harness gauntlet passed for {config_key}: "
                    f"eligible={result.metadata.get('promotion_eligible')}"
                )
                if result.metadata.get("promotion_eligible"):
                    self._consider_promotion(
                        config_key,
                        model_path,
                        result.metadata.get("win_rates", {}).get("heuristic", 0) / 100,
                        result.metadata.get("elo_delta", 0),
                    )
            else:
                logger.warning(
                    f"[EvaluationFeedback] Single-harness gauntlet failed for {config_key}: "
                    f"{result.error or 'unknown error'}"
                )
        except ImportError:
            logger.debug("[EvaluationFeedback] trigger_evaluation not available for fallback")
        except Exception as e:
            logger.error(f"[EvaluationFeedback] Single-harness fallback failed: {e}")

    async def _dispatch_gauntlet_to_work_queue(
        self, config_key: str, model_path: str, board_type: str, num_players: int
    ) -> None:
        """Dispatch gauntlet to work queue for cluster execution (coordinator nodes).

        January 28, 2026: Added to enable coordinator nodes to dispatch gauntlet
        evaluations to GPU cluster nodes instead of running them locally.
        This follows the same pattern as EvaluationDaemon._dispatch_gauntlet_to_cluster().

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            model_path: Path to model checkpoint
            board_type: Board type string
            num_players: Number of players
        """
        try:
            from app.coordination.work_distributor import (
                get_work_distributor,
                DistributedWorkConfig,
            )

            distributor = get_work_distributor()
            # Priority 85: Higher than selfplay (50) but lower than critical training (100)
            config = DistributedWorkConfig(priority=85, require_gpu=True)

            work_id = await distributor.submit_evaluation(
                candidate_model=model_path,
                baseline_model=None,
                games=200,  # Standard gauntlet size
                board=board_type,
                num_players=num_players,
                evaluation_type="gauntlet",
                config=config,
            )

            if work_id:
                logger.info(
                    f"[EvaluationFeedback] Gauntlet dispatched to cluster: {work_id} "
                    f"for {config_key} (model={model_path})"
                )
            else:
                logger.error(
                    f"[EvaluationFeedback] Failed to dispatch gauntlet to cluster for {config_key}"
                )

        except ImportError:
            logger.warning(
                "[EvaluationFeedback] WorkDistributor not available, cannot dispatch gauntlet"
            )
        except Exception as e:
            logger.error(f"[EvaluationFeedback] Gauntlet dispatch failed for {config_key}: {e}")

    def _trigger_gauntlet_all_baselines(self, config_key: str) -> None:
        """Trigger gauntlet evaluation against all baselines after regression.

        January 3, 2026 (Sprint 12 P1): When a major regression is detected,
        we trigger a comprehensive gauntlet evaluation against ALL baseline
        opponents (random, heuristic, and any other canonical models) to
        reassess the model's true strength.

        This helps detect if the regression was a fluke or if the model
        genuinely needs more training. Fresh Elo data guides curriculum
        and training decisions.
        """
        logger.info(
            f"[EvaluationFeedback] Triggering all-baseline gauntlet for {config_key} "
            f"(post-regression reassessment)"
        )

        try:
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.warning(f"[EvaluationFeedback] Invalid config_key format: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            # Find the current canonical model for this config
            try:
                from app.models.discovery import get_canonical_model_path

                model_path = get_canonical_model_path(board_type, num_players)
                if not model_path:
                    logger.debug(
                        f"[EvaluationFeedback] No canonical model found for {config_key}"
                    )
                    return
            except ImportError:
                logger.debug("[EvaluationFeedback] Model discovery not available")
                return

            # Use _trigger_evaluation which handles multi-harness gauntlet
            self._trigger_evaluation(config_key, str(model_path))

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"[EvaluationFeedback] Failed to trigger gauntlet: {e}")

    def _retry_evaluation(self, config_key: str, model_path: str, attempt: int) -> None:
        """Retry evaluation with modified parameters.

        P10-LOOP-2 (Dec 2025): Secondary retry logic for failed evaluations.
        Modifies parameters based on attempt number:
        - Attempt 1: Increase num_games by 50%
        - Attempt 2: Add delay and increase games by 100%
        - Attempt 3: Maximum games, longer delay
        """
        try:
            # Parse config key using canonical utility
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.debug(f"[EvaluationFeedback] Invalid config_key format: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            # Adjust parameters based on attempt
            base_games = 100
            delay = 0.0
            if attempt == 1:
                num_games = int(base_games * 1.5)  # 150 games
                delay = 2.0
            elif attempt == 2:
                num_games = int(base_games * 2.0)  # 200 games
                delay = 5.0
            else:
                num_games = int(base_games * 2.5)  # 250 games
                delay = 10.0

            async def _do_retry():
                if delay > 0:
                    await asyncio.sleep(delay)

                from app.coordination.pipeline_actions import trigger_evaluation

                logger.info(
                    f"[EvaluationFeedback] Retrying evaluation for {config_key}: "
                    f"num_games={num_games}, delay={delay}s"
                )

                result = await trigger_evaluation(
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=num_games,
                    max_retries=2,  # Reduced retries for secondary attempt
                )

                if result.success:
                    logger.info(f"[EvaluationFeedback] Secondary eval succeeded for {config_key}")
                else:
                    logger.warning(f"[EvaluationFeedback] Secondary eval failed for {config_key}")

            _safe_create_task(_do_retry(), f"retry_evaluation:{config_key}")

        except (AttributeError, TypeError, RuntimeError, asyncio.CancelledError) as e:
            logger.error(f"[EvaluationFeedback] Error setting up evaluation retry: {e}")


__all__ = ["EvaluationFeedbackMixin"]
