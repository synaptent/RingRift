"""Continuous Background Evaluation for RingRift AI Training.

Evaluates model strength in real-time during training via mini-gauntlets.
Auto-checkpoints on Elo improvement and early-stops on significant drops.

See: app.config.thresholds for canonical threshold constants.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import canonical Elo constants
try:
    from app.config.thresholds import INITIAL_ELO_RATING, ELO_DROP_ROLLBACK
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    ELO_DROP_ROLLBACK = 50.0

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for background evaluation.

    Note: Uses thresholds from app.config.thresholds.

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
    min_baseline_win_rates: Dict[str, float] = field(default_factory=lambda: {
        "random": 0.85,  # Must beat random at 85%+
        "heuristic": 0.60,  # Must beat heuristic at 60%+
    })


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
    """Runs continuous evaluation in a background thread."""

    def __init__(
        self,
        model_getter: Callable[[], Any],  # Function to get current model
        config: Optional[EvalConfig] = None,
    ):
        """Initialize background evaluator.

        Args:
            model_getter: Callable that returns current model state
            config: Evaluation configuration
        """
        self.model_getter = model_getter
        self.config = config or EvalConfig()

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

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir) if config else Path("data/eval_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        """Run a mini-gauntlet evaluation."""
        logger.info(f"[BackgroundEval] Running evaluation at step {step}")

        # Get current model
        model = self.model_getter()

        # Placeholder for actual gauntlet - would run games
        # This would integrate with baseline_gauntlet.py
        baseline_results = {}
        total_wins = 0
        total_games = 0

        for baseline in self.config.baselines:
            # Simulate games (replace with actual gauntlet)
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


def create_background_evaluator(
    model_getter: Callable[[], Any],
    eval_interval: int = 1000,
    games_per_eval: int = 20,
) -> BackgroundEvaluator:
    """Create a background evaluator."""
    config = EvalConfig(
        eval_interval_steps=eval_interval,
        games_per_eval=games_per_eval,
    )
    return BackgroundEvaluator(model_getter, config)
