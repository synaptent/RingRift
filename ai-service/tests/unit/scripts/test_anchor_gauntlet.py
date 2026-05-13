"""Unit tests for the fixed-checkpoint anchor gauntlet helpers."""

from __future__ import annotations

import math

import pytest

from scripts import run_anchor_gauntlet as gauntlet


def test_elo_diff_from_score_is_symmetric() -> None:
    assert gauntlet._elo_diff_from_score(0.5) == pytest.approx(0.0)
    assert gauntlet._elo_diff_from_score(0.75) == pytest.approx(190.8485, rel=1e-4)
    assert gauntlet._elo_diff_from_score(0.25) == pytest.approx(-190.8485, rel=1e-4)


def test_elo_diff_from_score_clamps_extreme_results() -> None:
    assert math.isfinite(gauntlet._elo_diff_from_score(1.0))
    assert gauntlet._elo_diff_from_score(1.0) == pytest.approx(
        -gauntlet._elo_diff_from_score(0.0)
    )


def test_wilson_interval_bounds_score() -> None:
    low, high = gauntlet._wilson_interval(33, 50)
    assert 0.5 < low < 0.66
    assert 0.66 < high < 0.8


def test_build_pairs_defaults_to_all_unique_combinations() -> None:
    participants = {
        "a": gauntlet.Participant("a", "model", "a.pth"),
        "b": gauntlet.Participant("b", "model", "b.pth"),
        "heuristic": gauntlet.Participant("heuristic", "heuristic"),
    }

    pairs = gauntlet._build_pairs(participants, [])

    assert [pair.key for pair in pairs] == [
        "a__vs__b",
        "a__vs__heuristic",
        "b__vs__heuristic",
    ]


def test_build_pairs_rejects_unknown_participant() -> None:
    participants = {"a": gauntlet.Participant("a", "model", "a.pth")}

    with pytest.raises(ValueError, match="unknown participant"):
        gauntlet._build_pairs(participants, [gauntlet.PairSpec("a", "missing")])


def test_solve_anchored_ratings_from_pair_deltas() -> None:
    pair_results = [
        {
            "a": "frontier",
            "b": "anchor",
            "elo_diff_a_minus_b": 200.0,
            "games_played": 400,
        },
        {
            "a": "replica",
            "b": "anchor",
            "elo_diff_a_minus_b": 100.0,
            "games_played": 400,
        },
    ]

    ratings = gauntlet._solve_anchored_ratings(
        ["frontier", "replica", "anchor"],
        pair_results,
        {"anchor": 1500.0},
    )

    assert ratings["anchor"] == pytest.approx(1500.0, abs=0.2)
    assert ratings["frontier"] == pytest.approx(1700.0, abs=0.5)
    assert ratings["replica"] == pytest.approx(1600.0, abs=0.5)


def test_dry_run_writes_plan_without_games(tmp_path) -> None:
    output = tmp_path / "plan.json"

    rc = gauntlet.main(
        [
            "--model",
            "frontier=/tmp/frontier.pth",
            "--baseline",
            "heuristic=heuristic",
            "--pair",
            "frontier:heuristic",
            "--fixed-rating",
            "heuristic=1500",
            "--games",
            "2",
            "--budget",
            "1",
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = output.read_text()
    assert "frontier__vs__heuristic" in payload
    assert '"pair_results_by_key": {}' in payload
    assert '"pair_results":' not in payload
