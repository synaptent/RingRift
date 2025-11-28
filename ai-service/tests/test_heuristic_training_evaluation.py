"""
Tests for the heuristic-training evaluation harness and HeuristicAI behaviour.

These tests are designed to guard against wiring/logic regressions in the
CMA-ES / GA fitness pipeline described in:

- scripts/run_cmaes_optimization.py
- scripts/run_genetic_heuristic_search.py

They focus on:

1. Verifying that the shared fitness evaluator meaningfully distinguishes
   clearly different heuristic weight profiles (baseline vs all-zero).
2. Verifying that HeuristicAI position evaluation on a nontrivial state
   actually depends on the applied weight vector (baseline vs all-zero).
"""

import json
import os
import sys
import tempfile
from typing import Dict, Any

# Ensure app package and training scripts are importable when running tests
# directly (mirrors pattern used in other ai-service tests).
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import (  # type: ignore  # noqa: E402
    AIConfig,
    BoardType,
    GamePhase,
    GameStatus,
    GameState,
)
from app.ai.heuristic_ai import HeuristicAI  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
)
from app.training.generate_data import (  # type: ignore  # noqa: E402
    create_initial_state,
)
from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    evaluate_fitness,
    create_heuristic_ai_with_weights,
)
from scripts.run_axis_aligned_tournament import (  # type: ignore  # noqa: E402
    load_axis_aligned_participants,
)
from scripts import (  # noqa: E402
    generate_axis_aligned_profiles,  # type: ignore[attr-defined]
)


def _make_zero_profile(baseline: Dict[str, float]) -> Dict[str, float]:
    """Return an all-zero weight profile matching the baseline schema."""
    return {k: 0.0 for k in baseline.keys()}


def test_heuristic_weight_keys_are_canonical_and_complete() -> None:
    """
    The canonical HEURISTIC_WEIGHT_KEYS list must:
    - Be non-empty and stable for external tooling.
    - Contain only keys from BASE_V1_BALANCED_WEIGHTS.
    - Cover all WEIGHT_* attributes exposed by HeuristicAI.
    """
    # 1. Canonical list is non-empty.
    assert isinstance(HEURISTIC_WEIGHT_KEYS, list)
    assert HEURISTIC_WEIGHT_KEYS, "HEURISTIC_WEIGHT_KEYS must not be empty"

    # 2. Keys match the BASE_V1_BALANCED_WEIGHTS schema (no extras/missing).
    baseline_keys = set(BASE_V1_BALANCED_WEIGHTS.keys())
    canonical_keys = set(HEURISTIC_WEIGHT_KEYS)

    # No key in the canonical ordering may fall outside the baseline schema.
    assert canonical_keys.issubset(baseline_keys)
    # And we expect a 1:1 correspondence between baseline schema and canonical
    # ordering so that optimisation/diagnostic tooling can rely on it.
    assert canonical_keys == baseline_keys

    # 3. Every WEIGHT_* attribute on HeuristicAI must be included.
    heuristic_ai_weight_attrs = {
        name
        for name in dir(HeuristicAI)
        if name.startswith("WEIGHT_")
    }
    assert heuristic_ai_weight_attrs.issubset(canonical_keys)


def test_evaluate_fitness_zero_profile_is_strictly_worse_than_baseline(
) -> None:
    """
    Baseline vs baseline must evaluate to ~0.5, while a degenerate "zero"
    profile must be clearly worse vs the same baseline under the shared
    fitness harness.

    This protects against wiring bugs where candidate and baseline weights
    are accidentally made identical, or where the candidate weights are
    never actually applied to the HeuristicAI instances.
    """
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    zero_profile = _make_zero_profile(baseline)

    baseline_stats: Dict[str, Any] = {}
    zero_stats: Dict[str, Any] = {}

    def _baseline_hook(d: Dict[str, Any]) -> None:
        baseline_stats.update(d)

    def _zero_hook(d: Dict[str, Any]) -> None:
        zero_stats.update(d)

    # Small evaluation budget to keep the test cheap but deterministic.
    games_per_eval = 8

    baseline_fitness = evaluate_fitness(
        candidate_weights=baseline,
        baseline_weights=baseline,
        games_per_eval=games_per_eval,
        board_type=BoardType.SQUARE8,
        verbose=False,
        opponent_mode="baseline-only",
        max_moves=200,
        debug_hook=_baseline_hook,
    )

    zero_fitness = evaluate_fitness(
        candidate_weights=zero_profile,
        baseline_weights=baseline,
        games_per_eval=games_per_eval,
        board_type=BoardType.SQUARE8,
        verbose=False,
        opponent_mode="baseline-only",
        max_moves=200,
        debug_hook=_zero_hook,
    )

    # Baseline vs baseline should be exactly symmetric under the alternating
    # colour schedule: expect a 0.5 win rate with mirrored W/L.
    assert abs(baseline_fitness - 0.5) < 1e-9
    assert (
        baseline_stats["wins"]
        + baseline_stats["draws"]
        + baseline_stats["losses"]
        == games_per_eval
    )
    assert baseline_stats["wins"] == baseline_stats["losses"]
    assert baseline_stats["draws"] == 0
    assert baseline_stats["weight_l2"] == 0.0

    # A clearly bad profile (all weights = 0) must lose heavily vs the
    # balanced baseline, and the L2 distance should be non-zero.
    assert zero_stats["wins"] == 0
    # Allow some slack but require that the zero profile be decisively worse
    # than 0.5 even at this small sample size.
    assert zero_stats["losses"] >= games_per_eval - 2
    assert zero_fitness < 0.25
    assert zero_stats["weight_l2"] > 0.0


def _build_midgame_state() -> GameState:
    """
    Construct a simple but nontrivial Square8 game state suitable for
    HeuristicAI evaluation tests.

    We start from the shared training initial-state helper to ensure that
    victory thresholds and per-player ring caps are aligned with the TS
    BOARD_CONFIGS, then let a baseline HeuristicAI play a few moves so that
    the board contains stacks and markers.
    """
    state = create_initial_state(BoardType.SQUARE8, num_players=2)
    assert state.game_status == GameStatus.ACTIVE
    assert state.current_phase == GamePhase.RING_PLACEMENT

    baseline_weights = dict(BASE_V1_BALANCED_WEIGHTS)
    ai = create_heuristic_ai_with_weights(
        player_number=1,
        weights=baseline_weights,
        difficulty=5,
    )

    # Play a handful of baseline moves to reach a mid-game-ish state.
    # We do not require a specific script, only that the resulting board
    # is non-empty so that heuristic features are exercised.
    max_plies = 6
    for _ in range(max_plies):
        if state.game_status != GameStatus.ACTIVE:
            break
        # Ensure the AI tracks the current side to move.
        ai.player_number = state.current_player
        move = ai.select_move(state)
        if move is None:
            # No legal moves: rely on the rules engine to mark a winner.
            break
        state = ai.rules_engine.apply_move(state, move)

    return state


def _make_heuristic_ai_with_profile(
    player_number: int,
    weights: Dict[str, float],
) -> HeuristicAI:
    """
    Helper that mirrors the training harness behaviour by constructing a
    HeuristicAI with a given weight dict applied on top of the internal
    profile mechanism.
    """
    config = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        rngSeed=123,
        heuristic_profile_id=None,
    )
    ai = HeuristicAI(player_number, config)
    for name, value in weights.items():
        setattr(ai, name, value)
    return ai


def test_heuristic_ai_position_evaluation_depends_on_weights() -> None:
    """
    On a mid-game state, HeuristicAI's scalar evaluation should differ
    between the balanced profile and an all-zero profile.

    This test focuses on the *position evaluation* (not full self-play)
    to ensure that the weight vector is actually consulted by the
    evaluation function used by select_move().
    """
    baseline_weights = dict(BASE_V1_BALANCED_WEIGHTS)
    zero_profile = _make_zero_profile(baseline_weights)

    state = _build_midgame_state()
    assert state.board.stacks  # We expect at least one stack on the board

    baseline_ai = _make_heuristic_ai_with_profile(
        player_number=1,
        weights=baseline_weights,
    )
    zero_ai = _make_heuristic_ai_with_profile(
        player_number=1,
        weights=zero_profile,
    )

    baseline_eval = baseline_ai.get_evaluation_breakdown(state)
    zero_eval = zero_ai.get_evaluation_breakdown(state)

    # The total evaluation score for the same player on the same state must
    # differ between the baseline and the degenerate zero-weight profile.
    # If this ever becomes equal, it strongly suggests that weights are not
    # being applied correctly or that evaluation is ignoring them.
    assert baseline_eval["total"] != zero_eval["total"]

    # As a sanity check, ensure both AIs are evaluating from the perspective
    # of the same player and that the game is still active.
    assert state.current_player in (1, 2)
    assert state.game_status == GameStatus.ACTIVE


def test_axis_aligned_profile_construction_single_factor() -> None:
    """
    Axis-aligned construction must:

    - Preserve the baseline weight schema.
    - Produce exactly one non-zero entry per profile.
    - Use a magnitude derived from the baseline (or 1.0 if baseline is 0).
    """
    # Use the first canonical key as a representative factor.
    factor_key = HEURISTIC_WEIGHT_KEYS[0]

    pos_profile, neg_profile, mag = (
        generate_axis_aligned_profiles.build_axis_aligned_profile(
            factor_key,
        )
    )

    baseline = BASE_V1_BALANCED_WEIGHTS
    baseline_keys = set(baseline.keys())

    # Profiles must have the same key set as the baseline.
    assert set(pos_profile.keys()) == baseline_keys
    assert set(neg_profile.keys()) == baseline_keys

    # Exactly one non-zero per profile, and it must match the requested factor.
    non_zero_pos = [k for k, v in pos_profile.items() if abs(v) > 1e-9]
    non_zero_neg = [k for k, v in neg_profile.items() if abs(v) > 1e-9]

    assert non_zero_pos == [factor_key]
    assert non_zero_neg == [factor_key]

    # Magnitude must follow the documented rule.
    base_val = float(baseline[factor_key])
    expected_mag = max(abs(base_val), 1.0)
    assert mag == expected_mag
    assert pos_profile[factor_key] == expected_mag
    assert neg_profile[factor_key] == -expected_mag


def test_axis_aligned_participant_loading_schema_and_metadata() -> None:
    """
    Axis-aligned participant loading must:

    - Discover JSON profiles under a directory.
    - Preserve the baseline weight schema.
    - Derive ids from ``meta.factor`` and ``meta.sign`` when present.
    - Preserve meta fields on the Participant.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_path = os.path.join(tmpdir, "WEIGHT_TERRITORY_pos.json")

        weights = dict(BASE_V1_BALANCED_WEIGHTS)
        meta = {
            "factor": "WEIGHT_TERRITORY",
            "sign": "pos",
            "base_magnitude": 3.5,
        }
        payload = {"weights": weights, "meta": meta}

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        participants = load_axis_aligned_participants(
            tmpdir,
            include_baseline=False,
        )

        # Exactly one non-baseline participant discovered.
        assert len(participants) == 1
        p = participants[0]

        # Id derived from factor + sign when available.
        assert p.id == "WEIGHT_TERRITORY_pos"

        # Weight schema matches the baseline definition.
        assert set(p.weights.keys()) == set(BASE_V1_BALANCED_WEIGHTS.keys())

        # Meta preserved verbatim and not marked as baseline.
        assert p.meta == meta
        assert not p.is_baseline

        # With include_baseline=True we should get an extra baseline entry.
        with_baseline = load_axis_aligned_participants(
            tmpdir,
            include_baseline=True,
        )
        assert len(with_baseline) == 2
        baseline = with_baseline[0]
        assert baseline.id == "baseline_v1_balanced"
        assert baseline.is_baseline
        assert baseline.meta.get("source") == "BASE_V1_BALANCED_WEIGHTS"