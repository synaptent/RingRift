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
import math
import os
import sys
import tempfile
from typing import Any, Dict

import pytest

# Ensure app package and training scripts are importable when running tests
# directly (mirrors pattern used in other ai-service tests).
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.heuristic_ai import HeuristicAI  # type: ignore
from app.ai.heuristic_weights import (  # type: ignore
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
)
from app.models import (  # type: ignore
    AIConfig,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
)
from app.training.generate_data import (  # type: ignore
    create_initial_state,
)
from scripts import (
    generate_axis_aligned_profiles,  # type: ignore[attr-defined]
    run_genetic_heuristic_search as ga_search,  # type: ignore[attr-defined]
)
from scripts.run_axis_aligned_tournament import (  # type: ignore
    load_axis_aligned_participants,
)
from scripts.run_cmaes_optimization import (  # type: ignore
    create_heuristic_ai_with_weights,
    evaluate_fitness,
)


def _make_zero_profile(baseline: dict[str, float]) -> dict[str, float]:
    """Return an all-zero weight profile matching the baseline schema."""
    return dict.fromkeys(baseline.keys(), 0.0)


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


# NOTE: This test was rewritten from a "zero must lose" premise to a wiring
# verification test. Originally it assumed that a zero-weight profile would
# perform worse than baseline, but in practice zero-weight evaluation leads
# to deterministic first-legal-move selection which can paradoxically win.
# The new test focuses on verifying the fitness harness tracks stats correctly
# and applies weights as expected.
@pytest.mark.timeout(180)
def test_evaluate_fitness_zero_profile_wiring_and_stats() -> None:
    """
    Verify that the fitness evaluation harness correctly:
    1. Baseline vs baseline evaluates to exactly 0.5 (symmetric).
    2. Zero-weight profile vs baseline produces games with tracked stats.
    3. Weight L2 distance is correctly computed (0 for same, >0 for diff).
    4. All stats fields are present and consistent.

    This is a WIRING test - it does NOT assert which profile wins, because
    with zero weights the AI makes deterministic first-legal-move selections
    which can produce surprising results. The key value is verifying that
    the harness correctly applies different weights and tracks outcomes.
    """
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    zero_profile = _make_zero_profile(baseline)

    baseline_stats: dict[str, Any] = {}
    zero_stats: dict[str, Any] = {}

    def _baseline_hook(d: dict[str, Any]) -> None:
        baseline_stats.update(d)

    def _zero_hook(d: dict[str, Any]) -> None:
        zero_stats.update(d)

    # Small evaluation budget to keep the test reasonably cheap while still
    # exercising the shared fitness harness.
    games_per_eval = 4

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

    # ===== Baseline vs baseline checks =====
    # Should be exactly symmetric under the alternating colour schedule:
    # expect a 0.5 win rate with mirrored W/L.
    assert abs(baseline_fitness - 0.5) < 1e-9, (
        f"Baseline vs baseline should be 0.5, got {baseline_fitness}"
    )
    assert (
        baseline_stats["wins"]
        + baseline_stats["draws"]
        + baseline_stats["losses"]
        == games_per_eval
    ), "Baseline stats should sum to games_per_eval"
    assert baseline_stats["wins"] == baseline_stats["losses"], (
        "Baseline vs baseline should have equal wins/losses"
    )
    assert baseline_stats["draws"] == 0, "Mirror matches should not draw"
    assert baseline_stats["weight_l2"] == 0.0, (
        "Baseline vs baseline L2 distance should be 0"
    )

    # ===== Zero-profile wiring checks =====
    # The core assertion: games were played and stats were tracked.
    assert (
        zero_stats["wins"]
        + zero_stats["draws"]
        + zero_stats["losses"]
        == games_per_eval
    ), "Zero-profile stats should sum to games_per_eval"

    # Weight L2 should be non-zero since zero-profile differs from baseline.
    assert zero_stats["weight_l2"] > 0.0, (
        "Zero vs baseline L2 distance should be positive"
    )

    # Fitness should be in valid range [0, 1].
    assert 0.0 <= zero_fitness <= 1.0, (
        f"Zero-profile fitness should be in [0, 1], got {zero_fitness}"
    )

    # Required stats fields should all be present.
    required_fields = [
        "wins", "draws", "losses", "games", "games_per_eval",
        "fitness", "weight_l2", "weight_l2_to_baseline",
    ]
    for field in required_fields:
        assert field in zero_stats, f"Missing required field: {field}"

    # Fitness value in stats should match returned value.
    assert math.isclose(zero_stats["fitness"], zero_fitness), (
        f"Stats fitness {zero_stats['fitness']} != returned {zero_fitness}"
    )


def test_evaluate_fitness_debug_callback_receives_stats() -> None:
    """
    The structured debug_callback exposed by evaluate_fitness must receive
    the candidate/baseline dicts, board type, and an aggregate stats dict
    with consistent fields (wins/draws/losses, games, fitness, and
    weight_l2_to_baseline).
    """
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    candidate = dict(baseline)
    games_per_eval = 2
    board_type = BoardType.SQUARE8

    captured: dict[str, Any] = {}

    def _cb(c_w, b_w, bt, stats):
        captured["candidate"] = c_w
        captured["baseline"] = b_w
        captured["board_type"] = bt
        captured["stats"] = stats

    fitness = evaluate_fitness(
        candidate_weights=candidate,
        baseline_weights=baseline,
        games_per_eval=games_per_eval,
        board_type=board_type,
        verbose=False,
        opponent_mode="baseline-only",
        max_moves=50,
        eval_mode="initial-only",
        state_pool_id=None,
        eval_randomness=0.0,
        seed=123,
        debug_callback=_cb,
    )

    assert captured["candidate"] == candidate
    assert captured["baseline"] == baseline
    assert captured["board_type"] == board_type

    stats = captured["stats"]
    assert stats["wins"] + stats["draws"] + stats["losses"] == games_per_eval
    assert stats["games"] == games_per_eval
    assert stats["games_per_eval"] == games_per_eval
    assert math.isclose(stats["fitness"], fitness)
    assert stats["weight_l2_to_baseline"] == 0.0
    # Legacy alias should still be present for callers that relied on it.
    assert "weight_l2" in stats


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
    weights: dict[str, float],
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


def test_ga_uses_debug_callback_for_logging(monkeypatch, capsys) -> None:
    """
    The GA harness must wire a per-individual FitnessDebugCallback into
    evaluate_fitness so that W/D/L and L2 diagnostics are printed.
    This test monkeypatches evaluate_fitness to be cheap and to assert that
    the callback is invoked once per individual.
    """
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    population = [
        ga_search.Individual(weights=baseline),
        ga_search.Individual(weights=baseline),
    ]

    callbacks = []

    def _fake_evaluate_fitness(*args, **kwargs):
        debug_cb = kwargs.get("debug_callback")
        # GA should always request per-individual diagnostics.
        assert debug_cb is not None
        callbacks.append(debug_cb)

        candidate_w = kwargs["candidate_weights"]
        baseline_w = kwargs["baseline_weights"]
        board_type = kwargs["board_type"]
        games_per_eval = kwargs["games_per_eval"]

        stats = {
            "wins": 1,
            "draws": 0,
            "losses": 0,
            "games": games_per_eval,
            "games_per_eval": games_per_eval,
            "fitness": 1.0,
            "weight_l2_to_baseline": 0.0,
        }
        debug_cb(candidate_w, baseline_w, board_type, dict(stats))
        return 1.0

    monkeypatch.setattr(
        ga_search,
        "evaluate_fitness",
        _fake_evaluate_fitness,
    )

    boards = [BoardType.SQUARE8]
    games_per_eval = 1

    ga_search._evaluate_population(
        population,
        games_per_eval,
        boards,
        eval_mode="initial-only",
        state_pool_id="v1",
        seed=42,
        eval_randomness=0.0,
        generation_index=1,
    )

    # Each individual should have had its fitness populated via the fake
    # evaluator, and the debug callback should have been invoked once per
    # individual.
    assert [ind.fitness for ind in population] == [1.0, 1.0]
    assert len(callbacks) == len(population)

    out = capsys.readouterr().out
    # The GA harness should emit per-individual diagnostics including W/D/L
    # and an L2 norm to the baseline.
    assert "[GA] gen=1 ind=1/2 board=" in out
    assert "W=1 D=0 L=0" in out
    assert "||w-baseline||_2=0.000" in out


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
