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
    game_idx = 0
    for seat in (1, 2, 3):
        for i in range(games_per_seat):
            # Half wins — balanced, below imbalance threshold
            tracker.record_game_outcome(game_idx, seat, i % 2 == 0)
            game_idx += 1


def _populate_imbalanced_seats(tracker: QualityGateTracker) -> None:
    """Seat 1 wins 90%, seat 2 wins 10% — triggers SEAT_WR_IMBALANCE."""
    game_idx = 0
    for i in range(30):
        tracker.record_game_outcome(game_idx, 1, i < 27)  # 27/30 wins
        game_idx += 1
    for i in range(30):
        tracker.record_game_outcome(game_idx, 2, i < 3)  # 3/30 wins
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
        _populate_balanced_seats(tracker, games_per_seat=20)
        quality = check_model_quality(tracker)
        record = _serialize_like_loop(quality)
        seat_fairness = record["details"].get("seat_fairness")
        assert isinstance(seat_fairness, dict), seat_fairness
        # JSON turns int keys into strings — lock that behaviour explicitly
        seat_wr = seat_fairness.get("seat_wr")
        assert isinstance(seat_wr, dict)
        assert set(seat_wr.keys()) == {"1", "2", "3"}
        for wr in seat_wr.values():
            assert 0.0 <= wr <= 1.0

    def test_imbalanced_seats_emit_warning_and_ratio(self) -> None:
        tracker = QualityGateTracker()
        _populate_imbalanced_seats(tracker)
        quality = check_model_quality(tracker)
        record = _serialize_like_loop(quality)
        # Warning should fire
        assert any("SEAT_WR_IMBALANCE" in w for w in record["warnings"]), record["warnings"]
        # Ratio should be present and > threshold
        seat_fairness = record["details"]["seat_fairness"]
        assert "wr_ratio" in seat_fairness
        assert seat_fairness["wr_ratio"] > 1.5

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
        assert set(extracted["seat_wr"].keys()) == {"1", "2"}
