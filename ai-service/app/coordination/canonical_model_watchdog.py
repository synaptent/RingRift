"""Canonical Model Watchdog — hourly verification that all models beat random.

Catches silent model corruption, encoding mismatches, and regression that
bypassed promotion gates. Any model that cannot beat random is functionally
broken and triggers an alert.

Mar 24, 2026: Created after discovering 10/12 canonical models were near-random
despite 171 training iterations. A simple "beats random" contract would have
caught this within 1 hour.
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional

from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

# Win rate thresholds by player count
WIN_THRESHOLDS = {2: 0.55, 3: 0.30, 4: 0.25}
GAMES_PER_CHECK = 10
MODELS_DIR = Path(os.environ.get("RINGRIFT_MODELS_DIR", "models"))


class CanonicalModelWatchdog(HandlerBase):
    """Periodically validates all canonical models can beat random."""

    _event_source = "CanonicalModelWatchdog"

    def __init__(self, cycle_interval: float = 3600.0):
        super().__init__(name="canonical_model_watchdog", cycle_interval=cycle_interval)
        self._last_results: dict[str, dict] = {}
        self._configs = [
            ("hex8", 2), ("hex8", 3), ("hex8", 4),
            ("square8", 2), ("square8", 3), ("square8", 4),
            ("square19", 2), ("square19", 3), ("square19", 4),
            ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
        ]

    def _get_event_subscriptions(self) -> dict:
        return {}

    async def _run_cycle(self) -> None:
        """Check all canonical models against random."""
        import asyncio
        results = await asyncio.to_thread(self._check_all_models)
        failures = [r for r in results if not r["passed"]]

        if failures:
            logger.warning(
                f"[ModelWatchdog] {len(failures)}/{len(results)} models FAILED "
                f"beats-random check: {[f['config'] for f in failures]}"
            )
            for f in failures:
                await self._safe_emit_event_async(
                    "CANONICAL_MODEL_DEGRADED",
                    {
                        "config_key": f["config"],
                        "win_rate": f["win_rate"],
                        "threshold": f["threshold"],
                        "error": f.get("error"),
                    },
                )
        else:
            logger.info(
                f"[ModelWatchdog] All {len(results)} models pass beats-random check"
            )

    def _check_all_models(self) -> list[dict]:
        """Run smoke test for each canonical model (blocking, runs in thread)."""
        results = []
        for board_type, num_players in self._configs:
            config_key = f"{board_type}_{num_players}p"
            model_path = MODELS_DIR / f"canonical_{config_key}.pth"

            if not model_path.exists():
                results.append({
                    "config": config_key, "passed": False,
                    "win_rate": 0, "threshold": WIN_THRESHOLDS.get(num_players, 0.5),
                    "error": "model file missing",
                })
                continue

            result = self._check_single_model(
                str(model_path), board_type, num_players, config_key
            )
            results.append(result)
            self._last_results[config_key] = result

        return results

    def _check_single_model(
        self, model_path: str, board_type: str, num_players: int, config_key: str
    ) -> dict:
        """Play GAMES_PER_CHECK games against random for one model."""
        threshold = WIN_THRESHOLDS.get(num_players, 0.5)
        try:
            from app.models import BoardType
            from app.training.game_gauntlet import (
                play_single_game, create_neural_ai, create_baseline_ai,
                BaselineOpponent,
            )

            bt = BoardType(board_type)
            ai = create_neural_ai(
                player=1, board_type=bt, model_path=model_path,
                num_players=num_players, use_search=False, temperature=0.1,
            )

            wins = 0
            for _ in range(GAMES_PER_CHECK):
                opp_ais = {
                    p: create_baseline_ai(
                        BaselineOpponent.RANDOM, p, bt, num_players=num_players,
                    )
                    for p in range(2, num_players + 1)
                }
                result = play_single_game(
                    candidate_ai=ai, opponent_ai=opp_ais[2],
                    board_type=bt, num_players=num_players,
                    candidate_player=1, max_moves=500,
                    opponent_ais=opp_ais if num_players > 2 else None,
                )
                if result.candidate_won:
                    wins += 1

            win_rate = wins / GAMES_PER_CHECK
            passed = win_rate >= threshold
            if not passed:
                logger.warning(
                    f"[ModelWatchdog] FAIL {config_key}: {wins}/{GAMES_PER_CHECK} "
                    f"({win_rate:.0%}) vs random, threshold={threshold:.0%}"
                )
            return {
                "config": config_key, "passed": passed,
                "win_rate": win_rate, "threshold": threshold,
                "wins": wins, "games": GAMES_PER_CHECK,
            }
        except Exception as e:
            logger.error(f"[ModelWatchdog] ERROR {config_key}: {e}")
            return {
                "config": config_key, "passed": False,
                "win_rate": 0, "threshold": threshold,
                "error": str(e)[:200],
            }

    def get_last_results(self) -> dict[str, dict]:
        """Get results from the most recent check cycle."""
        return dict(self._last_results)
