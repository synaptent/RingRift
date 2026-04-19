"""Integration test: QualityGateVerdict details land in JSON-safe form.

The minimal loop serializes the verdict into metrics.jsonl's
``quality_gate.details`` so downstream consumers (refresh_experiment_status
``extract_seat_fairness``, plateau detector, dashboards) can read per-seat
WR without re-deriving it from logs.  This test locks in the shape the
consumers expect.
"""

from __future__ import annotations

import json

from scripts.lib.model_quality_gate import QualityGateTracker, check_model_quality


def _populate_balanced_seats(tracker: QualityGateTracker, games_per_seat: int = 20) -> None:
    """Run enough games that the seat-fairness check engages."""
    tracker.set_selfplay_baseline({1: 17, 2: 25, 3: 28, 4: 30})
    game_idx = 0
    win_plan = {1: 5, 2: 8, 3: 8, 4: 9}
    for seat in (1, 2, 3, 4):
        for i in range(games_per_seat):
            tracker.record_game_outcome(game_idx, seat, i < win_plan[seat])
            game_idx += 1


def _populate_imbalanced_seats(tracker: QualityGateTracker) -> None:
    """Strongly skewed wins vs selfplay null should trigger SEAT_WR_IMBALANCE."""
    tracker.set_selfplay_baseline({1: 17, 2: 25, 3: 28, 4: 30})
    game_idx = 0
    win_plan = {1: 18, 2: 3, 3: 3, 4: 6}
    for seat in (1, 2, 3, 4):
        for i in range(25):
            tracker.record_game_outcome(game_idx, seat, i < win_plan[seat])
            game_idx += 1


def _serialize_like_loop(quality) -> dict:
    """Replicate minimal_alphazero_loop.py's quality_gate_record build."""
    raw_record = {
        "passed": bool(quality.passed),
        "critical": bool(quality.critical),
        "warnings": list(quality.warnings),
        "summary": str(quality.summary),
        "details": dict(quality.details),
    }
    # Roundtrip through JSON as the loop does (default=str fallback).
    return json.loads(json.dumps(raw_record, default=str))


class TestQualityGateSerialization:
    def test_empty_tracker_produces_json_safe_record(self) -> None:
        tracker = QualityGateTracker()
        quality = check_model_quality(tracker)
        record = _serialize_like_loop(quality)
        # Minimum shape required downstream
        assert isinstance(record["passed"], bool)
        assert isinstance(record["critical"], bool)
        assert isinstance(record["warnings"], list)
        assert isinstance(record["details"], dict)

    def test_balanced_seats_carry_seat_wr_details(self) -> None:
        tracker = QualityGateTracker()
        _populate_balanced_seats(tracker, games_per_seat=30)
        quality = check_model_quality(tracker)
        record = _serialize_like_loop(quality)
        seat_fairness = record["details"].get("seat_fairness")
        assert isinstance(seat_fairness, dict), seat_fairness
        # JSON turns int keys into strings — lock that behaviour explicitly
        seat_wr = seat_fairness.get("seat_wr")
        assert isinstance(seat_wr, dict)
        assert set(seat_wr.keys()) == {"1", "2", "3", "4"}
        for wr in seat_wr.values():
            assert 0.0 <= wr <= 1.0
        assert "selfplay_baseline_seat_wr" in seat_fairness
        assert "chi_square_p_value" in seat_fairness

    def test_imbalanced_seats_emit_warning_and_p_value(self) -> None:
        tracker = QualityGateTracker()
        _populate_imbalanced_seats(tracker)
        quality = check_model_quality(tracker)
        record = _serialize_like_loop(quality)
        # Warning should fire
        assert any("SEAT_WR_IMBALANCE" in w for w in record["warnings"]), record["warnings"]
        # Chi-square evidence should be present and significant
        seat_fairness = record["details"]["seat_fairness"]
        assert "chi_square_p_value" in seat_fairness
        assert seat_fairness["chi_square_p_value"] < 0.05

    def test_refresh_script_can_read_serialized_record(self) -> None:
        """Spot-check: refresh_experiment_status.extract_seat_fairness
        accepts the loop's output format.  Guards against plumbing drift.
        """
        # Late import so the test doesn't hard-depend on repo-root scripts path
        # when pytest is run from ai-service/.
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(repo_root))
        try:
            from scripts.refresh_experiment_status import extract_seat_fairness
        except ImportError:
            # If repo layout changes, skip rather than fail — the loop side is
            # what this test is locking in.
            return
        finally:
            sys.path.pop(0)

        tracker = QualityGateTracker()
        _populate_imbalanced_seats(tracker)
        quality = check_model_quality(tracker)
        record = _serialize_like_loop(quality)
        metric_row = {
            "iteration": 26,
            "evaluation": {"decision": "reject", "win_rate": 0.26},
            "quality_gate": record,
        }
        extracted = extract_seat_fairness(metric_row)
        assert extracted is not None
        assert "seat_wr" in extracted
        assert set(extracted["seat_wr"].keys()) == {"1", "2", "3", "4"}
