from __future__ import annotations

from scripts import run_self_play_soak as soak


def _rec(reason: str) -> soak.GameRecord:
    return soak.GameRecord(
        index=0,
        num_players=2,
        board_type="square8",
        engine_mode="heuristic-only",
        seed=0,
        length=1,
        status="completed",
        winner=1,
        termination_reason=reason,
    )


def test_fail_on_anomaly_flags_illegal_ai_and_phase_mismatches() -> None:
    reasons = [
        "ai_selected_illegal_move:place_ring",
        "illegal_move_for_phase:ring_placement:move_stack",
        "phase_move_mismatch:movement->no_line_action",
        "ai_move_not_forced_elimination",
        "recorded_player_mismatch",
        "illegal_moves_in_forced_elimination",
        "no_ai_for_current_player",
        "ai_returned_no_move",
        "step_exception:ValueError",
        "unknown",
    ]
    for reason in reasons:
        assert soak._has_anomalies([_rec(reason)])


def test_fail_on_anomaly_ignores_normal_terminations() -> None:
    reasons = [
        "max_moves_reached",
        "env_done_flag",
        "status:completed",
        "territory",
        "ring_elimination",
        "lps",
    ]
    for reason in reasons:
        assert not soak._has_anomalies([_rec(reason)])

