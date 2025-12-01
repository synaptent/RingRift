from __future__ import annotations

"""Tests for HeuristicAI evaluation modes (full vs light).

These tests verify:

- AIConfig.heuristic_eval_mode is normalised correctly by HeuristicAI.
- Tier-2 structural features are evaluated only in "full" mode and
  completely skipped in "light" mode.
- get_evaluation_breakdown remains consistent with evaluate_position
  and reports 0.0 for Tier-2 components in "light" mode.
- The CMA-ES / GA fitness harness applies the board-based mapping
  TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD when constructing HeuristicAI
  instances.
"""

import os
import sys
from typing import Dict, List

import pytest

# Ensure app and scripts packages are importable when running tests
# directly from the ai-service root.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    AIType,
    BoardType,
)
from app.ai.heuristic_ai import HeuristicAI  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
)
from app.training.generate_data import (  # type: ignore  # noqa: E402
    create_initial_state,
)
import scripts.run_cmaes_optimization as cmaes  # type: ignore  # noqa: E402
import scripts.run_self_play_soak as soak  # type: ignore  # noqa: E402


TIER2_KEYS: List[str] = [
    "line_potential",
    "line_connectivity",
    "territory_closure",
    "territory_safety",
    "vulnerability",
    "overtake_potential",
    "forced_elimination_risk",
    "lps_action_advantage",
]


def _make_active_state(board_type: BoardType = BoardType.SQUARE8):
    """Helper to construct a small, non-terminal GameState."""
    state = create_initial_state(board_type=board_type, num_players=2)
    assert state.game_status.value.lower() == "active"
    return state


def test_heuristic_eval_mode_default_and_override() -> None:
    """HeuristicAI should default to full mode and respect explicit light."""
    cfg_default = AIConfig(
        difficulty=5,
        randomness=None,
        rngSeed=None,
    )
    ai_default = HeuristicAI(player_number=1, config=cfg_default)
    assert ai_default.eval_mode == "full"

    cfg_light = AIConfig(
        difficulty=5,
        randomness=None,
        rngSeed=None,
        heuristic_eval_mode="light",
    )
    ai_light = HeuristicAI(player_number=1, config=cfg_light)
    assert ai_light.eval_mode == "light"

    # Unknown values must also normalise to full for safety.
    cfg_unknown = AIConfig(
        difficulty=5,
        randomness=None,
        rngSeed=None,
        heuristic_eval_mode="unknown",
    )
    ai_unknown = HeuristicAI(player_number=1, config=cfg_unknown)
    assert ai_unknown.eval_mode == "full"


def test_tier2_methods_called_in_full_mode_and_skipped_in_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier-2 structural features must be gated by eval_mode."""
    # Track call counts per Tier-2 method.
    call_counts: Dict[str, int] = {
        key: 0 for key in TIER2_KEYS
    }

    # Wrap each Tier-2 method to increment its counter.
    def _wrap(name: str):
        original = getattr(HeuristicAI, f"_evaluate_{name}")

        def _wrapped(self, game_state):
            call_counts[name] += 1
            return original(self, game_state)

        return _wrapped

    for key in TIER2_KEYS:
        monkeypatch.setattr(
            HeuristicAI,
            f"_evaluate_{key}",
            _wrap(key),
        )

    state = _make_active_state()

    # Full mode: Tier-2 methods should be called by both APIs.
    cfg_full = AIConfig(
        difficulty=5,
        randomness=None,
        rngSeed=None,
        heuristic_eval_mode="full",
    )
    ai_full = HeuristicAI(player_number=1, config=cfg_full)

    ai_full.evaluate_position(state)
    ai_full.get_evaluation_breakdown(state)

    for key in TIER2_KEYS:
        assert call_counts[key] > 0, f"{key} not called in full mode"

    # Reset counters and construct a light-mode AI under the same patches.
    for key in call_counts:
        call_counts[key] = 0

    cfg_light2 = AIConfig(
        difficulty=5,
        randomness=None,
        rngSeed=None,
        heuristic_eval_mode="light",
    )
    ai_light2 = HeuristicAI(player_number=1, config=cfg_light2)

    ai_light2.evaluate_position(state)
    ai_light2.get_evaluation_breakdown(state)

    for key in TIER2_KEYS:
        assert (
            call_counts[key] == 0
        ), f"{key} should not be called in light mode"


def test_breakdown_matches_total_and_zeroes_tier2_in_light() -> None:
    """In light mode, Tier-2 components must be exactly 0.0."""
    state = _make_active_state()
    cfg_light = AIConfig(
        difficulty=5,
        randomness=None,
        rngSeed=None,
        heuristic_eval_mode="light",
    )
    ai_light = HeuristicAI(player_number=1, config=cfg_light)

    total = ai_light.evaluate_position(state)
    breakdown = ai_light.get_evaluation_breakdown(state)

    assert pytest.approx(breakdown["total"]) == total

    for key in TIER2_KEYS:
        assert key in breakdown
        assert breakdown[key] == 0.0


@pytest.mark.parametrize(
    "board_type,expected_mode",
    [
        (BoardType.SQUARE8, "full"),
        (BoardType.SQUARE19, "light"),
        (BoardType.HEXAGONAL, "light"),
    ],
)
def test_evaluate_fitness_uses_board_based_heuristic_mode(
    board_type: BoardType,
    expected_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The CMA-ES/GA fitness harness must select heuristic_eval_mode based
    on board type using TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.
    """
    recorded_modes: List[str | None] = []

    original = cmaes.create_heuristic_ai_with_weights

    def _fake_play_single_game(*_args, **_kwargs) -> tuple[int, int]:
        """Fast stub for play_single_game used in tests."""
        return 0, 0

    def _fake_play_single_game_from_state(
        *_args, **_kwargs
    ) -> tuple[int, int]:
        """Fast stub for play_single_game_from_state used in tests."""
        return 0, 0

    monkeypatch.setattr(cmaes, "play_single_game", _fake_play_single_game)
    monkeypatch.setattr(
        cmaes,
        "play_single_game_from_state",
        _fake_play_single_game_from_state,
    )

    def _spy_create(
        player_number: int,
        weights: Dict[str, float],
        *,
        difficulty: int = 5,
        randomness: float = 0.0,
        rng_seed: int | None = None,
        heuristic_eval_mode: str | None = None,
    ) -> HeuristicAI:
        recorded_modes.append(heuristic_eval_mode)
        return original(
            player_number,
            weights,
            difficulty=difficulty,
            randomness=randomness,
            rng_seed=rng_seed,
            heuristic_eval_mode=heuristic_eval_mode,
        )

    monkeypatch.setattr(
        cmaes,
        "create_heuristic_ai_with_weights",
        _spy_create,
    )

    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    games_per_eval = 2

    # Use the same weights for candidate and baseline; we care only about
    # the heuristic_eval_mode passed into AIConfig.
    cmaes.evaluate_fitness(
        candidate_weights=baseline,
        baseline_weights=baseline,
        games_per_eval=games_per_eval,
        board_type=board_type,
        verbose=False,
    )

    # At least one HeuristicAI should have been constructed for this board.
    assert recorded_modes
    unique_modes = {m if m is not None else "full" for m in recorded_modes}
    assert unique_modes == {expected_mode}


@pytest.mark.parametrize(
    "board_type,expected_mode",
    [
        (BoardType.SQUARE8, "full"),
        (BoardType.SQUARE19, "light"),
        (BoardType.HEXAGONAL, "light"),
    ],
)
def test_soak_mixed_ai_pool_uses_board_based_heuristic_mode(
    board_type: BoardType,
    expected_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The Python self-play soak harness (_build_mixed_ai_pool) must also apply
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD when constructing HeuristicAI
    instances, mirroring the CMA-ES/GA fitness harness behaviour.
    """
    recorded_modes: List[str | None] = []

    def _fake_get_profile(_difficulty: int) -> Dict[str, Any]:
        # Force all sampled AIs to be Heuristic so we always exercise the
        # heuristic_eval_mode plumbing.
        return {
            "ai_type": AIType.HEURISTIC,
            "randomness": 0.0,
            "think_time_ms": 0,
            "profile_id": "v1-heuristic-5",
        }

    def _fake_create_ai_instance(
        ai_type: AIType,
        player_number: int,
        config: AIConfig,
    ) -> object:
        # Capture the heuristic_eval_mode passed to AIConfig for each player.
        recorded_modes.append(getattr(config, "heuristic_eval_mode", None))

        class _DummyAI:
            def __init__(self, player: int) -> None:
                self.player_number = player

        return _DummyAI(player_number)

    monkeypatch.setattr(soak, "_get_difficulty_profile", _fake_get_profile)
    monkeypatch.setattr(soak, "_create_ai_instance", _fake_create_ai_instance)

    pool = soak._build_mixed_ai_pool(
        game_index=0,
        player_numbers=[1, 2],
        engine_mode="mixed",
        base_seed=1234,
        difficulty_band="canonical",
        board_type=board_type,
    )

    # Sanity: we should have created at least one heuristic AI.
    assert pool
    assert recorded_modes

    unique_modes = {m if m is not None else "full" for m in recorded_modes}
    assert unique_modes == {expected_mode}
