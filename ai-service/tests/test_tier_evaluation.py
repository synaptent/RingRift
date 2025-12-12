#!/usr/bin/env python
"""Tests for tier evaluation config, runner, and CLI harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# Ensure app/ and scripts/ are importable when running tests directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from app.models import AIType, BoardType  # noqa: E402
from app.training.tier_eval_config import (  # noqa: E402
    TierEvaluationConfig,
    TierOpponentConfig,
    get_tier_config,
)
from app.training.tier_eval_runner import run_tier_evaluation  # noqa: E402
from app.training.eval_pools import (  # noqa: E402
    HEURISTIC_TIER_SPECS,
    POOL_PATHS,
    run_all_heuristic_tiers,
    run_heuristic_tier_eval,
)
from scripts import run_tier_gate  # noqa: E402


class TestTierEvaluationRunner:
    """Tests for the tier evaluation runner and config wiring."""

    def test_random_vs_random_basic_stats(self) -> None:
        """Random-vs-random smoke test for wiring and statistics."""
        tier_config = TierEvaluationConfig(
            tier_name="TEST",
            display_name="Test random vs random (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=4,
            candidate_difficulty=1,
            time_budget_ms=0,
            opponents=[
                TierOpponentConfig(
                    id="random_baseline",
                    description="Random baseline",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
            ],
            min_win_rate_vs_baseline=0.0,
            max_regression_vs_previous_tier=None,
            description="Unit-test tier profile for random-vs-random.",
        )

        result = run_tier_evaluation(
            tier_config=tier_config,
            candidate_id="random_candidate",
            seed=123,
            num_games_override=4,
        )

        assert result.tier_name == "TEST"
        assert result.total_games == 4
        assert len(result.matchups) == 1

        matchup = result.matchups[0]
        assert matchup.opponent_id == "random_baseline"
        assert matchup.games == 4
        assert (
            matchup.wins + matchup.losses + matchup.draws == matchup.games
        )
        assert matchup.average_game_length > 0.0

        # Victory reasons should be populated from RingRiftEnv.info
        assert matchup.victory_reasons
        assert sum(matchup.victory_reasons.values()) == matchup.games

        # Gating metrics and criteria should be present and JSON-serialisable
        data = result.to_dict()
        assert data["tier"] == "TEST"
        assert data["candidate"]["id"] == "random_candidate"
        assert "stats" in data
        assert "by_opponent" in data["stats"]
        assert "random_baseline" in data["stats"]["by_opponent"]
        assert (
            data["stats"]["by_opponent"]["random_baseline"]["games"] == 4
        )

        # With min_win_rate_vs_baseline == 0.0 this criterion should pass
        assert result.criteria["min_win_rate_vs_baseline"] is True
        # No previous-tier constraint configured for this synthetic tier
        assert (
            result.criteria["no_major_regression_vs_previous_tier"]
            is None
        )


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "tier_name",
    [
        "D2_SQ19_2P",
        "D2_SQ8_3P",
    ],
)
def test_run_tier_evaluation_multiboard_and_multiplayer_smoke(
    tier_name: str,
) -> None:
    """Smoke test run_tier_evaluation on multi-board / multiplayer tiers."""
    tier_cfg = get_tier_config(tier_name)
    candidate_id = f"test_candidate_{tier_name.lower()}"
    # Keep this smoke test fast: we only need to validate wiring + JSON shape.
    num_games_override = 2

    result = run_tier_evaluation(
        tier_config=tier_cfg,
        candidate_id=candidate_id,
        seed=1,
        num_games_override=num_games_override,
        # Keep the smoke test runtime bounded on large boards (square19/hex)
        # where fully-random play can otherwise run for thousands of moves.
        max_moves_override=100,
    )

    assert result.tier_name == tier_cfg.tier_name
    assert result.board_type == tier_cfg.board_type
    assert result.num_players == tier_cfg.num_players
    assert result.candidate_difficulty == tier_cfg.candidate_difficulty
    assert result.total_games == num_games_override * len(
        tier_cfg.opponents
    )
    assert len(result.matchups) == len(tier_cfg.opponents)

    data = result.to_dict()
    assert data["tier"] == tier_name
    assert data["candidate"]["id"] == candidate_id
    assert "metrics" in data
    assert "criteria" in data
    assert "win_rate_vs_baseline" in data["metrics"]
    assert "win_rate_vs_previous_tier" in data["metrics"]
    assert "min_win_rate_vs_baseline" in data["criteria"]
    assert "no_major_regression_vs_previous_tier" in data["criteria"]


class TestTierEvaluationCli:
    """CLI smoke test for run_tier_evaluation.py."""

    @pytest.mark.slow
    def test_run_tier_evaluation_cli_smoke(self, tmp_path) -> None:
        """Run a tiny D2 evaluation via the CLI and check JSON output."""
        output_path = tmp_path / "tier_eval_d2.json"
        cmd = [
            sys.executable,
            "scripts/run_tier_evaluation.py",
            "--tier",
            "D2",
            "--candidate-config",
            "baseline_smoke",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--output-json",
            str(output_path),
        ]

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        assert proc.returncode == 0, proc.stderr
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["tier"] == "D2"
        assert "overall_pass" in data
        assert isinstance(data["overall_pass"], bool)


def test_heuristic_tier_eval_smoke(monkeypatch) -> None:
    """Smoke test for heuristic tier eval harness on eval pools.

    Uses a stubbed load_state_pool so the test does not depend on on-disk
    JSONL eval pools and runs quickly.
    """
    # Ensure we have at least one heuristic tier defined.
    assert HEURISTIC_TIER_SPECS, (
        "Expected at least one heuristic heuristic tier spec"
    )
    tier = HEURISTIC_TIER_SPECS[0]

    # Stub load_state_pool to return a single valid GameState snapshot.
    from app.training import eval_pools as eval_pools_mod  # noqa: E402
    from app.training.generate_data import (  # noqa: E402
        create_initial_state,
    )

    def _fake_load_state_pool(
        board_type,
        pool_id="v1",
        max_states=None,
        num_players=None,
    ):
        state = create_initial_state(
            board_type=board_type,
            num_players=num_players or tier.num_players,
        )
        return [state]

    monkeypatch.setattr(
        eval_pools_mod,
        "load_state_pool",
        _fake_load_state_pool,
    )

    # Run a tiny single-game eval for the first tier.
    result = run_heuristic_tier_eval(
        tier_spec=tier,
        rng_seed=123,
        max_games=1,
        max_moves_override=40,
    )

    assert result["tier_id"] == tier.id
    assert result["games_played"] == 1
    assert set(result["results"].keys()) == {"wins", "losses", "draws"}
    assert "ring_margin_mean" in result["margins"]
    assert "territory_margin_mean" in result["margins"]
    assert "mean" in result["latency_ms"]
    assert "p95" in result["latency_ms"]

    # Also exercise the multi-tier wrapper with a single-tier filter so the
    # top-level report structure is validated.
    report = run_all_heuristic_tiers(
        tiers=HEURISTIC_TIER_SPECS,
        rng_seed=321,
        max_games=1,
        max_moves_override=40,
        tier_ids=[tier.id],
    )
    assert "run_id" in report
    assert "timestamp" in report
    assert "tiers" in report
    assert report["tiers"]
    assert len(report["tiers"]) == 1
    tier_entry = report["tiers"][0]
    assert tier_entry["tier_id"] == tier.id


def test_heuristic_tier_specs_have_pool_paths() -> None:
    """Ensure every heuristic tier spec has a configured eval pool path.

    This keeps HEURISTIC_TIER_SPECS and POOL_PATHS in sync so that adding a
    new tier without wiring an eval pool fails fast in tests instead of at
    runtime on long jobs.
    """
    assert HEURISTIC_TIER_SPECS, (
        "Expected at least one heuristic tier spec"
    )

    missing: list[str] = []
    for spec in HEURISTIC_TIER_SPECS:
        key = (spec.board_type, spec.eval_pool_id)
        if key not in POOL_PATHS:
            missing.append(
                f"{spec.id} -> ({spec.board_type!r}, {spec.eval_pool_id!r})"
            )

    assert not missing, (
        "One or more heuristic tier specs reference eval pools that are not "
        f"configured in POOL_PATHS: {', '.join(missing)}"
    )


def test_run_tier_gate_cli_smoke(monkeypatch, capsys) -> None:
    """Smoke test for the heuristic tier gate CLI wrapper."""
    # Use the first heuristic tier spec as the target.
    assert HEURISTIC_TIER_SPECS, (
        "Expected at least one heuristic heuristic tier spec"
    )
    tier = HEURISTIC_TIER_SPECS[0]

    # Stub run_heuristic_tier_eval so the test does not depend on
    # on-disk eval pools.
    def _fake_run_heuristic_tier_eval(
        tier_spec,
        rng_seed,
        max_games=None,
        max_moves_override=None,
        **_kwargs,
    ):
        assert tier_spec.id == tier.id
        assert isinstance(rng_seed, int)
        assert max_games in (None, 1)
        assert max_moves_override in (None, 40)
        return {
            "tier_id": tier_spec.id,
            "tier_name": tier_spec.name,
            "board_type": tier_spec.board_type.value,
            "num_players": tier_spec.num_players,
            "eval_pool_id": tier_spec.eval_pool_id,
            "candidate_profile_id": tier_spec.candidate_profile_id,
            "baseline_profile_id": tier_spec.baseline_profile_id,
            "games_requested": tier_spec.num_games,
            "games_played": 1,
            "results": {"wins": 1, "losses": 0, "draws": 0},
            "margins": {
                "ring_margin_mean": 1.0,
                "territory_margin_mean": 0.0,
            },
            "latency_ms": {"mean": 0.5, "p95": 1.0},
            "total_moves": 10,
            "victory_reasons": {"ring_elimination": 1},
        }

    monkeypatch.setattr(
        run_tier_gate,
        "run_heuristic_tier_eval",
        _fake_run_heuristic_tier_eval,
    )

    # Run the CLI-style main with a temporary argv.
    import sys as _sys

    argv_backup = list(_sys.argv)
    _sys.argv = [
        "run_tier_gate.py",
        "--tier-id",
        tier.id,
        "--seed",
        "42",
        "--max-games",
        "1",
    ]
    try:
        run_tier_gate.main()
    finally:
        _sys.argv = argv_backup

    captured = capsys.readouterr()
    # The CLI prints a single JSON object to stdout; parse and sanity-check it.
    payload = json.loads(captured.out)
    assert payload["tier_id"] == tier.id
    assert payload["tier_name"] == tier.name
    assert payload["games_played"] == 1

def test_run_tier_gate_cli_multiboard_sq19_smoke(tmp_path) -> None:
    """Smoke test multiboard difficulty-tier mode of run_tier_gate CLI."""
    output_path = tmp_path / "tier_gate_D2_SQ19_2P.json"
    promo_path = tmp_path / "promotion_D2_SQ19_2P.json"
    cmd = [
        sys.executable,
        "scripts/run_tier_gate.py",
        "--tier",
        "D2_SQ19_2P",
        "--candidate-model-id",
        "test_candidate_sq19_d2",
        "--num-games",
        "1",
        "--seed",
        "1",
        "--time-budget-ms",
        "0",
        "--max-moves",
        "40",
        "--output-json",
        str(output_path),
        "--promotion-plan-out",
        str(promo_path),
    ]

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert output_path.exists()
    assert promo_path.exists()

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["tier"] == "D2_SQ19_2P"

    plan = json.loads(promo_path.read_text(encoding="utf-8"))
    assert plan["tier"] == "D2_SQ19_2P"
    assert plan["candidate_model_id"] == "test_candidate_sq19_d2"
    assert plan["decision"] in ("promote", "reject")
