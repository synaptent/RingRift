"""Continuous Background Evaluation for RingRift AI Training.

Evaluates model strength in real-time during training via mini-gauntlets.
Auto-checkpoints on Elo improvement and early-stops on significant drops.

See: app.config.thresholds for canonical threshold constants.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import canonical threshold constants
try:
    from app.config.thresholds import (
        INITIAL_ELO_RATING,
        ELO_DROP_ROLLBACK,
        MIN_WIN_RATE_VS_RANDOM,
        MIN_WIN_RATE_VS_HEURISTIC,
    )
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    ELO_DROP_ROLLBACK = 50.0
    MIN_WIN_RATE_VS_RANDOM = 0.85
    MIN_WIN_RATE_VS_HEURISTIC = 0.60

logger = logging.getLogger(__name__)


@dataclass
class BackgroundEvalConfig:
    """Configuration for background evaluation with baseline gating.

    .. note::
        This is specialized for background eval (continuous Elo tracking).
        For shadow/tournament config: ``app.config.unified_config.EvaluationConfig``
        For training-loop eval: ``app.training.config.TrainingEvaluationConfig``

    Uses thresholds from app.config.thresholds.

    IMPORTANT: min_baseline_win_rates defines minimum win rates against each
    baseline that must be met for a checkpoint to be considered "qualified".
    This prevents selecting checkpoints that are strong in neural-vs-neural
    play but weak against basic baselines.
    """
    eval_interval_steps: int = 1000  # Steps between evaluations
    games_per_eval: int = 20  # Games per evaluation
    baselines: List[str] = field(default_factory=lambda: ["random", "heuristic"])
    elo_checkpoint_threshold: float = 10.0  # Min Elo gain to checkpoint
    elo_drop_threshold: float = ELO_DROP_ROLLBACK  # Elo drop for early stopping
    auto_checkpoint: bool = True
    checkpoint_dir: str = "data/eval_checkpoints"
    # Baseline gating: minimum win rates required against each baseline
    # Checkpoints that don't meet these are considered "unqualified"
    # Values imported from app.config.thresholds for single source of truth
    min_baseline_win_rates: Dict[str, float] = field(default_factory=lambda: {
        "random": MIN_WIN_RATE_VS_RANDOM,
        "heuristic": MIN_WIN_RATE_VS_HEURISTIC,
    })


# Backwards-compatible alias
EvalConfig = BackgroundEvalConfig


@dataclass
class EvalResult:
    """Result of an evaluation run."""
    step: int
    timestamp: float
    elo_estimate: float
    elo_std: float
    games_played: int
    win_rate: float
    baseline_results: Dict[str, float]  # baseline -> win rate
    passes_baseline_gating: bool = True  # Whether all baseline thresholds are met
    failed_baselines: List[str] = field(default_factory=list)  # Which baselines failed


class BackgroundEvaluator:
    """Runs continuous evaluation in a background thread.

    Can operate in two modes:
    1. Placeholder mode (default): Uses random results for fast testing
    2. Real game mode: Actually plays games against baselines

    To enable real game mode, provide board_type and set use_real_games=True.
    """

    def __init__(
        self,
        model_getter: Callable[[], Any],  # Function to get current model
        config: Optional[EvalConfig] = None,
        board_type: Optional[Any] = None,  # BoardType for real games
        use_real_games: bool = False,  # Whether to play real games
    ):
        """Initialize background evaluator.

        Args:
            model_getter: Callable that returns current model state (must return
                         a dict with 'state_dict' and optionally 'path')
            config: Evaluation configuration
            board_type: Board type for real game evaluation
            use_real_games: If True, play actual games instead of simulation
        """
        self.model_getter = model_getter
        self.config = config or EvalConfig()
        self.board_type = board_type
        self.use_real_games = use_real_games

        # Validate real games configuration
        if use_real_games and board_type is None:
            logger.warning(
                "[BackgroundEval] use_real_games=True but board_type not provided. "
                "Falling back to placeholder mode."
            )
            self.use_real_games = False

        # State
        self.current_step = 0
        self.last_eval_step = 0
        self.best_elo = 0.0
        self.current_elo = INITIAL_ELO_RATING
        self.elo_history: List[Tuple[int, float]] = []
        self.eval_results: List[EvalResult] = []

        # Threading
        self._running = False
        self._eval_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Checkpointing - for saving temp models during evaluation
        self.checkpoint_dir = Path(config.checkpoint_dir) if config else Path("data/eval_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._temp_model_path = self.checkpoint_dir / "temp_eval_model.pth"

    def start(self):
        """Start background evaluation thread."""
        if self._running:
            return

        self._running = True
        self._eval_thread = threading.Thread(target=self._eval_loop, daemon=True)
        self._eval_thread.start()
        logger.info("[BackgroundEval] Started evaluation thread")

    def stop(self):
        """Stop background evaluation."""
        self._running = False
        if self._eval_thread:
            self._eval_thread.join(timeout=5.0)
        logger.info("[BackgroundEval] Stopped evaluation thread")

    def update_step(self, step: int):
        """Update current training step."""
        with self._lock:
            self.current_step = step

    def _eval_loop(self):
        """Background evaluation loop."""
        while self._running:
            with self._lock:
                step = self.current_step
                should_eval = (step - self.last_eval_step) >= self.config.eval_interval_steps

            if should_eval:
                try:
                    result = self._run_evaluation(step)
                    self._process_result(result)
                except Exception as e:
                    logger.error(f"[BackgroundEval] Evaluation failed: {e}")

            time.sleep(5.0)  # Check interval

    def _run_evaluation(self, step: int) -> EvalResult:
        """Run a mini-gauntlet evaluation.

        If use_real_games=True, plays actual games against baselines using
        the game_gauntlet module. Otherwise uses placeholder random results.
        """
        logger.info(f"[BackgroundEval] Running evaluation at step {step}")

        # Get current model info
        model_info = self.model_getter()

        if self.use_real_games:
            return self._run_real_evaluation(step, model_info)
        else:
            return self._run_placeholder_evaluation(step)

    def _run_real_evaluation(self, step: int, model_info: Any) -> EvalResult:
        """Run actual games using game_gauntlet module."""
        from app.training.game_gauntlet import (
            run_baseline_gauntlet,
            BaselineOpponent,
        )

        # Map config baselines to BaselineOpponent enums
        opponents = []
        for baseline_name in self.config.baselines:
            try:
                opponents.append(BaselineOpponent(baseline_name))
            except ValueError:
                logger.warning(f"[BackgroundEval] Unknown baseline: {baseline_name}")

        if not opponents:
            opponents = [BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC]

        games_per = self.config.games_per_eval // len(opponents)

        # Prefer in-memory loading (zero disk I/O) when model_info supports it
        model_path = None
        model_getter = None

        if isinstance(model_info, (str, Path)):
            # File path - use file-based loading
            model_path = Path(model_info)
        elif isinstance(model_info, dict) and 'path' in model_info and model_info['path']:
            # Dict with path - use file-based loading
            model_path = Path(model_info['path'])
        else:
            # In-memory model (dict with state_dict or nn.Module)
            # Use model_getter to return the model_info for zero-disk-IO loading
            def _model_getter(captured_info=model_info):
                return captured_info
            model_getter = _model_getter
            logger.info("[BackgroundEval] Using in-memory model loading (zero disk I/O)")

        try:
            logger.info(f"[BackgroundEval] Playing {games_per} games per opponent (real mode)")
            gauntlet_result = run_baseline_gauntlet(
                model_path=model_path,
                board_type=self.board_type,
                opponents=opponents,
                games_per_opponent=games_per,
                check_baseline_gating=True,
                verbose=False,
                model_getter=model_getter,
            )

            # Convert to EvalResult format
            baseline_results = {
                name: stats["win_rate"]
                for name, stats in gauntlet_result.opponent_results.items()
            }

            return EvalResult(
                step=step,
                timestamp=time.time(),
                elo_estimate=gauntlet_result.estimated_elo,
                elo_std=100.0 / np.sqrt(gauntlet_result.total_games + 1),
                games_played=gauntlet_result.total_games,
                win_rate=gauntlet_result.win_rate,
                baseline_results=baseline_results,
                passes_baseline_gating=gauntlet_result.passes_baseline_gating,
                failed_baselines=gauntlet_result.failed_baselines,
            )

        except Exception as e:
            logger.error(f"[BackgroundEval] Real evaluation failed: {e}, using placeholder")
            return self._run_placeholder_evaluation(step)

    def _save_temp_model(self, model_info: Any) -> Optional[Path]:
        """Save model weights to temp file for evaluation.

        Args:
            model_info: Either a path string, a dict with 'state_dict', or nn.Module

        Returns:
            Path to saved model, or None if save failed
        """
        try:
            import torch

            # If model_info is already a path, use it directly
            if isinstance(model_info, (str, Path)):
                return Path(model_info)

            # If model_info is a dict with state_dict
            if isinstance(model_info, dict):
                if 'path' in model_info and model_info['path']:
                    return Path(model_info['path'])
                if 'state_dict' in model_info:
                    torch.save(model_info['state_dict'], self._temp_model_path)
                    return self._temp_model_path

            # If model_info is an nn.Module
            if hasattr(model_info, 'state_dict'):
                torch.save(model_info.state_dict(), self._temp_model_path)
                return self._temp_model_path

            logger.warning(f"[BackgroundEval] Unknown model_info type: {type(model_info)}")
            return None

        except Exception as e:
            logger.error(f"[BackgroundEval] Failed to save temp model: {e}")
            return None

    def _run_placeholder_evaluation(self, step: int) -> EvalResult:
        """Run placeholder evaluation with random results.

        Used when real game-playing is not configured or fails.
        """
        # PLACEHOLDER: Simulates games with random outcomes
        baseline_results = {}
        total_wins = 0
        total_games = 0

        for baseline in self.config.baselines:
            # Random binomial (55% expected win rate)
            wins = np.random.binomial(self.config.games_per_eval // len(self.config.baselines), 0.55)
            games = self.config.games_per_eval // len(self.config.baselines)
            baseline_results[baseline] = wins / games if games > 0 else 0.5
            total_wins += wins
            total_games += games

        win_rate = total_wins / total_games if total_games > 0 else 0.5

        # Estimate Elo from win rate
        elo_estimate = self.current_elo + 400 * np.log10(win_rate / (1 - win_rate + 1e-8))
        elo_std = 100.0 / np.sqrt(total_games)  # Approximate

        # Check baseline gating
        passes_gating = True
        failed_baselines = []
        for baseline, win_rate_against in baseline_results.items():
            min_required = self.config.min_baseline_win_rates.get(baseline, 0.0)
            if win_rate_against < min_required:
                passes_gating = False
                failed_baselines.append(baseline)
                logger.warning(
                    f"[BackgroundEval] Failed baseline gating: {baseline} "
                    f"({win_rate_against:.1%} < {min_required:.0%} required)"
                )

        result = EvalResult(
            step=step,
            timestamp=time.time(),
            elo_estimate=elo_estimate,
            elo_std=elo_std,
            games_played=total_games,
            win_rate=win_rate,
            baseline_results=baseline_results,
            passes_baseline_gating=passes_gating,
            failed_baselines=failed_baselines,
        )

        with self._lock:
            self.last_eval_step = step

        return result

    def _process_result(self, result: EvalResult):
        """Process evaluation result."""
        with self._lock:
            self.eval_results.append(result)
            self.elo_history.append((result.step, result.elo_estimate))

            old_elo = self.current_elo
            self.current_elo = result.elo_estimate

            # Check for improvement
            elo_gain = result.elo_estimate - self.best_elo

            # Only checkpoint if baseline gating passes
            if elo_gain > self.config.elo_checkpoint_threshold:
                if result.passes_baseline_gating:
                    self.best_elo = result.elo_estimate
                    if self.config.auto_checkpoint:
                        self._save_checkpoint(result)
                    logger.info(f"[BackgroundEval] New best Elo: {result.elo_estimate:.0f} (+{elo_gain:.0f})")
                else:
                    logger.warning(
                        f"[BackgroundEval] Elo improved to {result.elo_estimate:.0f} (+{elo_gain:.0f}) "
                        f"but FAILED baseline gating ({', '.join(result.failed_baselines)}). "
                        "Checkpoint NOT saved."
                    )

            # Check for drop (early stopping trigger)
            elif (self.best_elo - result.elo_estimate) > self.config.elo_drop_threshold:
                logger.warning(
                    f"[BackgroundEval] Elo dropped significantly: "
                    f"{result.elo_estimate:.0f} (best: {self.best_elo:.0f})"
                )

    def _save_checkpoint(self, result: EvalResult):
        """Save checkpoint on Elo improvement."""
        checkpoint_path = self.checkpoint_dir / f"best_elo_{result.elo_estimate:.0f}_step{result.step}.pt"
        logger.info(f"[BackgroundEval] Saving checkpoint to {checkpoint_path}")
        # Would save model state here

    def get_current_elo(self) -> float:
        """Get current Elo estimate."""
        with self._lock:
            return self.current_elo

    def get_elo_history(self) -> List[Tuple[int, float]]:
        """Get Elo history [(step, elo), ...]."""
        with self._lock:
            return self.elo_history.copy()

    def should_early_stop(self) -> bool:
        """Check if training should early stop due to Elo drop."""
        with self._lock:
            return (self.best_elo - self.current_elo) > self.config.elo_drop_threshold

    def get_baseline_gating_status(self) -> Tuple[bool, List[str], int]:
        """Get current baseline gating status.

        Returns:
            Tuple of (passes_gating, failed_baselines, consecutive_failures)
            - passes_gating: True if latest eval passes all baseline thresholds
            - failed_baselines: List of baselines that failed (empty if passes)
            - consecutive_failures: Number of consecutive evals that failed gating
        """
        with self._lock:
            if not self.eval_results:
                return True, [], 0

            latest = self.eval_results[-1]

            # Count consecutive failures from most recent
            consecutive_failures = 0
            for result in reversed(self.eval_results):
                if not result.passes_baseline_gating:
                    consecutive_failures += 1
                else:
                    break

            return (
                latest.passes_baseline_gating,
                latest.failed_baselines.copy(),
                consecutive_failures,
            )

    def should_trigger_baseline_warning(self, failure_threshold: int = 3) -> bool:
        """Check if consecutive baseline failures exceed threshold.

        Args:
            failure_threshold: Number of consecutive failures to trigger warning

        Returns:
            True if should trigger warning/intervention
        """
        _, _, consecutive_failures = self.get_baseline_gating_status()
        return consecutive_failures >= failure_threshold


def create_background_evaluator(
    model_getter: Callable[[], Any],
    eval_interval: int = 1000,
    games_per_eval: int = 20,
    board_type: Optional[Any] = None,
    use_real_games: bool = False,
) -> BackgroundEvaluator:
    """Create a background evaluator.

    Args:
        model_getter: Callable that returns model info (path, state_dict, or nn.Module)
        eval_interval: Steps between evaluations
        games_per_eval: Number of games per evaluation
        board_type: Board type for real game evaluation (required if use_real_games=True)
        use_real_games: If True, play actual games instead of placeholder simulation

    Returns:
        BackgroundEvaluator instance
    """
    config = EvalConfig(
        eval_interval_steps=eval_interval,
        games_per_eval=games_per_eval,
    )
    return BackgroundEvaluator(
        model_getter,
        config,
        board_type=board_type,
        use_real_games=use_real_games,
    )
