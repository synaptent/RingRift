"""Tests for evaluation pools and the run_eval_tournaments demo harness."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import pytest

# Ensure `app.*` / `scripts.*` imports resolve when running pytest from
# ai-service/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from app.training.eval_pools import (  # noqa: E402
    POOL_PATHS,
    EvalPoolConfig,
    get_eval_pool_config,
    load_eval_pool,
)
import scripts.run_eval_tournaments as eval_tournaments  # noqa: E402


TEST_TIMEOUT_SECONDS = 60


# Subset of pools we expect to exist in this repo. Hex pools are wired in
# code but their JSONL files are generated out-of-band and may be absent,
# so they are intentionally excluded here.
NAMED_POOLS_UNDER_TEST: list[str] = [
    "square8_2p_core",
    "square19_2p_core",
    "square8_3p_baseline",
    "square8_4p_baseline",
    "square19_3p_baseline",
    "square19_4p_baseline",
]


def _skip_if_pool_file_missing(cfg: EvalPoolConfig) -> None:
    """Skip the current test if the underlying pool file does not exist."""
    key = (cfg.board_type, cfg.pool_id)
    path = POOL_PATHS.get(key)
    if not path or not os.path.exists(path):
        pytest.skip(
            "Eval pool file not found for "
            f"{cfg.name!r} at {path!r}. See the README under "
            "ai-service/data/eval_pools/ for generation commands."
        )


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
@pytest.mark.parametrize("pool_name", NAMED_POOLS_UNDER_TEST)
def test_eval_pools_load_and_validate(pool_name: str) -> None:
    """Basic integrity checks for named evaluation pools.

    For each configured pool we ensure that:

    - At least one scenario can be loaded via load_eval_pool.
    - Scenario board_type and num_players match the EvalPoolConfig.
    - The initial_state is accepted by DefaultRulesEngine.get_valid_moves
      without raising (we do not require non-empty move lists).
    """
    cfg = get_eval_pool_config(pool_name)
    _skip_if_pool_file_missing(cfg)

    scenarios = load_eval_pool(pool_name, max_scenarios=2)
    assert scenarios, f"Pool {pool_name!r} should contain at least 1 scenario"

    engine = DefaultRulesEngine()

    for scenario in scenarios:
        assert scenario.board_type == cfg.board_type
        assert scenario.num_players == cfg.num_players

        # Engine-level legality check: get_valid_moves must not raise.
        engine.get_valid_moves(
            scenario.initial_state,
            scenario.initial_state.current_player,
        )


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_run_eval_tournaments_demo_square8_3p(tmp_path: Path) -> None:
    """Demo tournament on the Square8 3-player pool.

    Uses a tiny configuration:

    - engine=random to avoid heavy search.
    - --num-games=1 and --demo to cap runtime.
    """
    pool_name = "square8_3p_baseline"
    cfg = get_eval_pool_config(pool_name)
    _skip_if_pool_file_missing(cfg)

    run_dir = tmp_path / "sq8_3p_demo"

    argv = [
        "--pool",
        pool_name,
        "--engine",
        "random",
        "--num-games",
        "1",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "123",
    ]

    rc = eval_tournaments.main(argv)
    assert rc == 0

    reports = list(run_dir.glob("*.json"))
    assert reports, "Expected a JSON report file in run-dir"
    report_path = reports[0]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pool_name"] == pool_name
    assert report["board_type"] == cfg.board_type.value
    assert report["num_players"] == cfg.num_players
    assert report["mode"] == "demo"
    assert report["total_scenarios"] >= 1
    assert report["total_games"] >= 1

    results = report["results"]
    assert isinstance(results, list) and results
    first = results[0]
    assert "scenario_id" in first
    assert "games_played" in first
    assert first["games_played"] >= 1
    assert "wins_by_player" in first
    assert "victory_reasons" in first


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_run_eval_tournaments_demo_square19_2p(tmp_path: Path) -> None:
    """Demo tournament on the Square19 2-player core pool.

    This exercises the large-board path while staying CI-safe:

    - engine=random (no heavy minimax/MCTS search).
    - --num-games=1 and --demo to limit the number of plies.
    """
    pool_name = "square19_2p_core"
    cfg = get_eval_pool_config(pool_name)
    _skip_if_pool_file_missing(cfg)

    run_dir = tmp_path / "sq19_2p_demo"

    argv = [
        "--pool",
        pool_name,
        "--engine",
        "random",
        "--num-games",
        "1",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "321",
    ]

    rc = eval_tournaments.main(argv)
    assert rc == 0

    reports = list(run_dir.glob("*.json"))
    assert reports, "Expected a JSON report file in run-dir"
    report_path = reports[0]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pool_name"] == pool_name
    assert report["board_type"] == cfg.board_type.value
    assert report["num_players"] == cfg.num_players
    assert report["mode"] == "demo"
    assert report["total_scenarios"] >= 1
    assert report["total_games"] >= 1

    results = report["results"]
    assert isinstance(results, list) and results
    first = results[0]
    assert "scenario_id" in first
    assert "games_played" in first
    assert first["games_played"] >= 1
    assert "wins_by_player" in first
    assert "victory_reasons" in first