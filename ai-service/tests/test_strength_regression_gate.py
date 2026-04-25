from __future__ import annotations

import json

from scripts import run_strength_regression_gate as gate  # type: ignore[import]


def test_compute_gate_without_significance_uses_raw_win_rate() -> None:
    threshold = gate.GateThreshold(
        min_win_rate=0.55,
        require_significance=False,
        confidence=0.95,
    )
    summary = gate._compute_gate(wins=6, losses=4, draws=0, threshold=threshold)
    assert summary["win_rate"] == 0.6
    assert summary["passes"] is True


def test_compute_gate_with_significance_requires_ci_lower_bound() -> None:
    threshold = gate.GateThreshold(
        min_win_rate=0.55,
        require_significance=True,
        confidence=0.95,
    )
    # 6/10 wins meets raw win-rate threshold but should fail the significance
    # gate because the Wilson lower bound remains below 0.55 for n=10.
    summary = gate._compute_gate(wins=6, losses=4, draws=0, threshold=threshold)
    assert summary["win_rate"] == 0.6
    assert summary["win_rate_ci_low"] is not None
    assert summary["win_rate_ci_low"] < 0.55
    assert summary["passes"] is False


def test_compute_gate_all_draws_fails_significance_gate() -> None:
    threshold = gate.GateThreshold(
        min_win_rate=0.52,
        require_significance=True,
        confidence=0.95,
    )
    summary = gate._compute_gate(wins=0, losses=0, draws=10, threshold=threshold)
    assert summary["decisive_games"] == 0
    assert summary["win_rate"] == 0.5
    assert summary["passes"] is False


def test_ci_mode_skips_blocking_matchup_when_all_eval_pools_missing(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        gate,
        "_default_matchups",
        lambda _mode: [
            gate.MatchupSpec(
                name="heuristic_vs_random",
                candidate="heuristic",
                opponent="random",
                pools=["missing_pool"],
                max_scenarios=1,
                games_per_scenario=1,
                max_moves=20,
                think_time_ms=1,
                use_neural_net=False,
                heuristic_move_sample_limit=None,
                heuristic_eval_mode=None,
                threshold=gate.GateThreshold(
                    min_win_rate=0.55,
                    require_significance=False,
                    confidence=0.95,
                ),
                blocking=True,
            )
        ],
    )
    monkeypatch.setattr(gate, "_pool_exists", lambda _pool_name: False)

    output = tmp_path / "report.json"
    code = gate.main(["--mode", "ci", "--output-json", str(output)])
    report = json.loads(output.read_text())

    assert code == 0
    matchup = report["matchups"]["heuristic_vs_random"]
    assert matchup["passes"] is True
    assert matchup["skipped"] is True
    assert matchup["skip_reason"] == "no_eval_pools_available"
    assert report["overall_pass"] is True


def test_nightly_mode_fails_blocking_matchup_when_all_eval_pools_missing(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        gate,
        "_default_matchups",
        lambda _mode: [
            gate.MatchupSpec(
                name="heuristic_vs_random",
                candidate="heuristic",
                opponent="random",
                pools=["missing_pool"],
                max_scenarios=1,
                games_per_scenario=1,
                max_moves=20,
                think_time_ms=1,
                use_neural_net=False,
                heuristic_move_sample_limit=None,
                heuristic_eval_mode=None,
                threshold=gate.GateThreshold(
                    min_win_rate=0.55,
                    require_significance=False,
                    confidence=0.95,
                ),
                blocking=True,
            )
        ],
    )
    monkeypatch.setattr(gate, "_pool_exists", lambda _pool_name: False)

    output = tmp_path / "report.json"
    code = gate.main(["--mode", "nightly", "--output-json", str(output)])
    report = json.loads(output.read_text())

    assert code == 1
    matchup = report["matchups"]["heuristic_vs_random"]
    assert matchup["passes"] is False
    assert matchup["skipped"] is True
    assert matchup["skip_reason"] == "no_eval_pools_available"
    assert report["overall_pass"] is False
