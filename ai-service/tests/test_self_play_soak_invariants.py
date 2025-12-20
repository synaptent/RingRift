import json
import os
import sys
from typing import Dict, List, Tuple

import pytest

# Ensure app and scripts packages are importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import (  # type: ignore  # noqa: E402
    BoardType,
    GamePhase,
    GameStatus,
)
from app.metrics import (  # type: ignore  # noqa: E402
    PYTHON_INVARIANT_VIOLATIONS,
)
import scripts.run_self_play_soak as soak  # type: ignore  # noqa: E402


def test_summarise_aggregates_invariant_counts() -> None:
    records: list[soak.GameRecord] = [
        soak.GameRecord(
            index=0,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=10,
            status="completed",
            winner=1,
            termination_reason="status:finished",
            invariant_violations_by_type={"S_INVARIANT_DECREASED": 1},
        ),
        soak.GameRecord(
            index=1,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=12,
            status="completed",
            winner=2,
            termination_reason="max_moves_reached",
            invariant_violations_by_type={"ACTIVE_NO_MOVES": 2},
        ),
    ]

    samples: list[dict[str, object]] = [{"dummy": True}]
    summary = soak._summarise(records, samples)

    assert summary["total_games"] == 2
    assert summary["completed_games"] == 1
    assert summary["max_moves_games"] == 1
    by_id = summary["invariant_violations_by_id"]
    assert by_id["INV-S-MONOTONIC"] == 1
    assert by_id["INV-ACTIVE-NO-MOVES"] == 2
    assert summary["invariant_violations_total"] == 3
    assert "invariant_violation_samples" in summary
    assert summary["invariant_violation_samples"] == samples


def test_summarise_includes_swap_rule_metrics() -> None:
    """Summary should always include pie-rule (swap) usage metrics.

    The summary JSON must expose:
    - swap_sides_total_moves: raw count of SWAP_SIDES moves across all games.
    - swap_sides_games: number of games where swap was used at least once.
    - swap_sides_games_fraction: fraction ∈ [0, 1] of games with swap.
    - avg_swap_sides_moves_per_game: average swap moves per game (≤1 for
      correct pie-rule semantics, but may be >1 if multiple swaps somehow
      occurred due to bugs).
    """
    records: list[soak.GameRecord] = [
        soak.GameRecord(
            index=0,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=10,
            status="completed",
            winner=1,
            termination_reason="status:finished",
            invariant_violations_by_type={},
            swap_sides_moves=1,
            used_pie_rule=True,
        ),
        soak.GameRecord(
            index=1,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=12,
            status="completed",
            winner=2,
            termination_reason="status:finished",
            invariant_violations_by_type={},
            swap_sides_moves=0,
            used_pie_rule=False,
        ),
        soak.GameRecord(
            index=2,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=8,
            status="completed",
            winner=1,
            termination_reason="status:finished",
            invariant_violations_by_type={},
            swap_sides_moves=1,
            used_pie_rule=True,
        ),
    ]

    summary = soak._summarise(records, None)

    # Raw aggregate fields
    assert summary["swap_sides_total_moves"] == 2
    assert summary["swap_sides_games"] == 2

    # Derived metrics
    assert "swap_sides_games_fraction" in summary
    assert summary["swap_sides_games_fraction"] == pytest.approx(2 / 3)

    assert "avg_swap_sides_moves_per_game" in summary
    assert summary["avg_swap_sides_moves_per_game"] == pytest.approx(2 / 3)


def test_summarise_swap_metrics_with_zero_games() -> None:
    """Swap metrics should be 0 when there are no games."""
    records: list[soak.GameRecord] = []
    summary = soak._summarise(records, None)

    assert summary["swap_sides_total_moves"] == 0
    assert summary["swap_sides_games"] == 0
    assert summary["swap_sides_games_fraction"] == 0.0
    assert summary["avg_swap_sides_moves_per_game"] == 0.0


def test_record_invariant_violation_counts_and_samples_capped() -> None:
    class DummyState:
        def __init__(self) -> None:
            self.board_type = BoardType.SQUARE8
            self.game_status = GameStatus.ACTIVE
            self.current_player = 1
            self.current_phase = GamePhase.RING_PLACEMENT

    state = DummyState()
    per_game: dict[str, int] = {}
    samples: list[dict[str, object]] = []

    for i in range(soak.MAX_INVARIANT_VIOLATION_SAMPLES + 5):
        soak._record_invariant_violation(  # type: ignore[arg-type]
            "S_INVARIANT_DECREASED",
            state,
            game_index=0,
            move_index=i,
            per_game_counts=per_game,
            samples=samples,
            prev_snapshot={"S": i},
            curr_snapshot={"S": i - 1},
        )

    assert per_game["S_INVARIANT_DECREASED"] == (
        soak.MAX_INVARIANT_VIOLATION_SAMPLES + 5
    )
    assert len(samples) == soak.MAX_INVARIANT_VIOLATION_SAMPLES
    first = samples[0]
    assert first["type"] == "S_INVARIANT_DECREASED"
    assert first["invariant_id"] == "INV-S-MONOTONIC"


def test_record_invariant_violation_emits_python_metrics() -> None:
    class DummyState:
        def __init__(self) -> None:
            self.board_type = BoardType.SQUARE8
            self.game_status = GameStatus.ACTIVE
            self.current_player = 1
            self.current_phase = GamePhase.RING_PLACEMENT

    state = DummyState()
    per_game: dict[str, int] = {}
    samples: list[dict[str, object]] = []

    soak._record_invariant_violation(  # type: ignore[arg-type]
        "S_INVARIANT_DECREASED",
        state,
        game_index=0,
        move_index=0,
        per_game_counts=per_game,
        samples=samples,
        prev_snapshot={"S": 10},
        curr_snapshot={"S": 9},
    )

    # Ensure the Prometheus counter has at least one sample with the
    # expected invariant_id/type label pair.
    collected = list(PYTHON_INVARIANT_VIOLATIONS.collect())
    assert collected, "No metrics collected for PYTHON_INVARIANT_VIOLATIONS"
    samples_for_metric = collected[0].samples
    matching = [
        s
        for s in samples_for_metric
        if s.labels.get("invariant_id") == "INV-S-MONOTONIC"
        and s.labels.get("type") == "S_INVARIANT_DECREASED"
    ]
    assert matching, (
        "Expected invariant violation metric sample was not "
        "recorded"
    )


def test_has_anomalies_detects_invariant_and_engine_anomalies() -> None:
    """_has_anomalies should only flag hard invariant/engine anomalies."""
    normal_records: list[soak.GameRecord] = [
        soak.GameRecord(
            index=0,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=10,
            status="completed",
            winner=1,
            termination_reason="status:completed",  # Fixed: was 'status:finished'
            invariant_violations_by_type={},
        ),
        soak.GameRecord(
            index=1,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=200,
            status="completed",
            winner=None,
            termination_reason="max_moves_reached",
            invariant_violations_by_type={},
        ),
    ]

    anomalous_records: list[soak.GameRecord] = [
        soak.GameRecord(
            index=2,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=5,
            status="completed",
            winner=None,
            termination_reason="no_legal_moves_for_current_player",
            invariant_violations_by_type={},
        ),
        soak.GameRecord(
            index=3,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=3,
            status="completed",
            winner=None,
            termination_reason="step_exception:RuntimeError",
            invariant_violations_by_type={},
        ),
        soak.GameRecord(
            index=4,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=2,
            status="completed",
            winner=None,
            termination_reason="error_reset:GameStateDeserialisation",
            invariant_violations_by_type={},
        ),
    ]

    # Purely normal terminations should not be treated as anomalies.
    assert soak._has_anomalies(normal_records) is False

    # Any hard invariant/engine anomaly should flip the flag.
    assert soak._has_anomalies(anomalous_records) is True


def test_python_strict_profile_overrides_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Verify python-strict profile sets expected configuration values."""
    captured_args: dict[str, object] = {}

    def fake_run(
        args: object,
    ) -> tuple[list[soak.GameRecord], list[dict[str, object]]]:
        captured_args["args"] = args
        return [], []

    monkeypatch.setattr(soak, "run_self_play_soak", fake_run)
    monkeypatch.setattr(soak, "_has_anomalies", lambda records: False)

    log_path = tmp_path / "soak.jsonl"

    argv = [
        "run_self_play_soak.py",
        "--log-jsonl",
        str(log_path),
        "--profile",
        "python-strict",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    soak.main()

    args = captured_args.get("args")
    assert args is not None
    assert getattr(args, "num_games") == 6
    assert getattr(args, "board_type") == "square8"
    assert getattr(args, "engine_mode") == "mixed"
    assert getattr(args, "num_players") == 2
    assert getattr(args, "max_moves") == 10000  # Updated from 150
    assert getattr(args, "gc_interval") == 5  # Default gc_interval (only set to 10 if explicitly 0)
    assert getattr(args, "profile") == "python-strict"


def test_build_healthcheck_summary_includes_invariants_and_samples() -> None:
    records: list[soak.GameRecord] = [
        soak.GameRecord(
            index=0,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=10,
            status="completed",
            winner=1,
            termination_reason="status:finished",
            invariant_violations_by_type={"S_INVARIANT_DECREASED": 1},
        ),
        soak.GameRecord(
            index=1,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=12,
            status="completed",
            winner=2,
            termination_reason="max_moves_reached",
            invariant_violations_by_type={"ACTIVE_NO_MOVES": 2},
        ),
    ]

    samples: list[dict[str, object]] = [
        {
            "type": "S_INVARIANT_DECREASED",
            "invariant_id": "INV-S-MONOTONIC",
            "game_index": 0,
            "move_index": 5,
            "board_type": "square8",
            "game_status": "active",
            "current_player": 1,
            "current_phase": "movement",
        },
    ]

    summary = soak._build_healthcheck_summary(  # type: ignore[attr-defined]
        profile="ai-healthcheck",
        board_types=["square8", "square19"],
        engine_pairs=["mixed_(light)_2p"],
        records=records,
        invariant_samples=samples,
    )

    assert summary["profile"] == "ai-healthcheck"
    assert set(summary["board_types"]) == {"square8", "square19"}
    assert summary["engine_pairs"] == ["mixed_(light)_2p"]

    by_id = summary["invariant_violations_by_id"]
    assert by_id["INV-S-MONOTONIC"] == 1
    assert by_id["INV-ACTIVE-NO-MOVES"] == 2
    assert summary["invariant_violations_total"] == 3

    # Samples should be exposed under both the original and the shorthand key.
    assert "invariant_violation_samples" in summary
    assert "samples" in summary
    assert summary["samples"][0]["type"] == "S_INVARIANT_DECREASED"
    assert summary["samples"][0]["invariant_id"] == "INV-S-MONOTONIC"

    # Parity placeholders must always be present for the health-check profile.
    assert "parity_mismatches" in summary
    assert summary["parity_mismatches"]["hash"] == 0
    assert summary["parity_mismatches"]["status"] == 0
    assert "parity_notes" in summary


def test_ai_healthcheck_profile_cli_writes_summary_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured_args: dict[str, object] = {}

    def fake_run_healthcheck(
        args: object,
    ) -> tuple[list[soak.GameRecord], dict[str, object]]:
        captured_args["args"] = args
        rec = soak.GameRecord(
            index=0,
            num_players=2,
            board_type="square8",
            engine_mode="mixed",
            seed=None,
            length=5,
            status="completed",
            winner=1,
            termination_reason="status:finished",
            invariant_violations_by_type={},
        )
        summary: dict[str, object] = {
            "profile": "ai-healthcheck",
            "board_types": ["square8", "square19", "hexagonal"],
            "engine_pairs": ["mixed_(light)_2p"],
            "total_games": 1,
            "completed_games": 1,
            "max_moves_games": 0,
            "invariant_violations_total": 0,
            "invariant_violations_by_id": {},
        }
        return [rec], summary

    monkeypatch.setattr(
        soak,
        "run_ai_healthcheck_profile",
        fake_run_healthcheck,
    )
    monkeypatch.setattr(soak, "_has_anomalies", lambda records: False)

    log_path = tmp_path / "healthcheck.log.jsonl"
    summary_path = tmp_path / "healthcheck.summary.json"

    argv = [
        "run_self_play_soak.py",
        "--log-jsonl",
        str(log_path),
        "--summary-json",
        str(summary_path),
        "--profile",
        "ai-healthcheck",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    soak.main()

    # Ensure the health-check profile was wired through to the helper.
    args = captured_args.get("args")
    assert args is not None
    assert getattr(args, "profile") == "ai-healthcheck"

    # Summary JSON should have been written and contain invariant keys.
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["profile"] == "ai-healthcheck"
    assert "invariant_violations_by_id" in payload
    assert "invariant_violations_total" in payload


# (duplicate test_* definitions removed; single copies remain above)
