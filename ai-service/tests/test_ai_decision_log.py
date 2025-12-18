"""Tests for AI Decision Logging module.

Tests the AIDecisionLog dataclass and associated utilities:
- Data serialization (to_dict, to_json)
- Context manager for decision tracking
- Integration with metrics and tracing
"""

import json
import time
from datetime import datetime, timezone

import pytest

from app.ai.decision_log import (
    AIDecisionLog,
    AIDecisionContext,
    log_ai_decision,
    track_ai_decision,
    create_decision_log_from_stats,
)


class TestAIDecisionLog:
    """Tests for AIDecisionLog dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        log = AIDecisionLog()
        assert log.game_id == ""
        assert log.move_number == 0
        assert log.difficulty == 0
        assert log.engine_type == ""
        assert log.time_ms == 0.0
        assert log.cache_hit is False
        assert log.used_fallback is False
        assert log.error is None
        assert isinstance(log.timestamp, datetime)

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        log = AIDecisionLog(
            game_id="game_123",
            move_number=5,
            difficulty=3,
            engine_type="mcts",
            simulations=800,
            time_ms=150.5,
            chosen_move="place(3,4)",
            move_score=0.65,
        )

        assert log.game_id == "game_123"
        assert log.move_number == 5
        assert log.difficulty == 3
        assert log.engine_type == "mcts"
        assert log.simulations == 800
        assert log.time_ms == 150.5
        assert log.chosen_move == "place(3,4)"
        assert log.move_score == 0.65

    def test_to_dict(self):
        """to_dict should return serializable dictionary."""
        log = AIDecisionLog(
            game_id="test",
            engine_type="minimax",
            search_depth=5,
        )

        data = log.to_dict()

        assert isinstance(data, dict)
        assert data['game_id'] == "test"
        assert data['engine_type'] == "minimax"
        assert data['search_depth'] == 5
        # Timestamp should be ISO format string
        assert isinstance(data['timestamp'], str)

    def test_to_json(self):
        """to_json should return valid JSON string."""
        log = AIDecisionLog(
            game_id="json_test",
            move_score=0.75,
        )

        json_str = log.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['game_id'] == "json_test"
        assert parsed['move_score'] == 0.75

    def test_to_structured_log(self):
        """to_structured_log should include event type."""
        log = AIDecisionLog(game_id="struct_test")
        structured = log.to_structured_log()

        assert structured['event'] == "ai_decision"
        assert structured['level'] == "info"
        assert structured['game_id'] == "struct_test"

    def test_to_structured_log_with_error(self):
        """to_structured_log should set level to error when error present."""
        log = AIDecisionLog(error="Something went wrong")
        structured = log.to_structured_log()

        assert structured['level'] == "error"
        assert structured['error'] == "Something went wrong"

    def test_summary_basic(self):
        """summary should return human-readable string."""
        log = AIDecisionLog(
            engine_type="mcts",
            chosen_move="place(1,2)",
            move_score=0.5,
            time_ms=100.0,
        )

        summary = log.summary()

        assert "[mcts]" in summary
        assert "move=place(1,2)" in summary
        assert "score=0.500" in summary
        assert "time=100.0ms" in summary

    def test_summary_with_simulations(self):
        """summary should include simulations when present."""
        log = AIDecisionLog(
            engine_type="mcts",
            chosen_move="move",
            move_score=0.5,
            time_ms=100.0,
            simulations=800,
        )

        summary = log.summary()
        assert "sims=800" in summary

    def test_summary_with_cache_hit(self):
        """summary should include cache info when present."""
        log = AIDecisionLog(
            engine_type="minimax",
            chosen_move="move",
            move_score=0.5,
            time_ms=50.0,
            cache_hit=True,
            cache_type="transposition",
        )

        summary = log.summary()
        assert "cache=transposition" in summary

    def test_summary_with_fallback(self):
        """summary should include fallback info when present."""
        log = AIDecisionLog(
            engine_type="mcts",
            chosen_move="move",
            move_score=0.0,
            time_ms=5000.0,
            used_fallback=True,
            fallback_reason="timeout",
        )

        summary = log.summary()
        assert "fallback=timeout" in summary


class TestAIDecisionContext:
    """Tests for AIDecisionContext context manager."""

    def test_basic_context_manager(self):
        """Context manager should track time automatically."""
        with AIDecisionContext(
            game_id="ctx_test",
            engine_type="test",
            auto_log=False,
        ) as ctx:
            time.sleep(0.01)  # 10ms

        # Time should be tracked
        assert ctx.decision.time_ms >= 10.0

    def test_record_move(self):
        """record_move should update decision."""
        with AIDecisionContext(auto_log=False) as ctx:
            ctx.record_move("place(3,4)", score=0.8, confidence=0.9)

        assert ctx.decision.chosen_move == "place(3,4)"
        assert ctx.decision.move_score == 0.8
        assert ctx.decision.move_confidence == 0.9

    def test_record_move_with_alternatives(self):
        """record_move should store alternatives."""
        with AIDecisionContext(auto_log=False) as ctx:
            ctx.record_move(
                "best_move",
                score=0.9,
                alternatives=[("alt1", 0.8), ("alt2", 0.7)],
            )

        assert len(ctx.decision.top_alternatives) == 2
        assert ctx.decision.top_alternatives[0] == ("alt1", 0.8)

    def test_record_search_stats(self):
        """record_search_stats should update decision."""
        with AIDecisionContext(auto_log=False) as ctx:
            ctx.record_search_stats(
                depth=5,
                simulations=800,
                nodes=10000,
                positions=5000,
            )

        assert ctx.decision.search_depth == 5
        assert ctx.decision.simulations == 800
        assert ctx.decision.nodes_evaluated == 10000
        assert ctx.decision.positions_searched == 5000

    def test_record_cache_hit(self):
        """record_cache_hit should update decision."""
        with AIDecisionContext(auto_log=False) as ctx:
            ctx.record_cache_hit("transposition")

        assert ctx.decision.cache_hit is True
        assert ctx.decision.cache_type == "transposition"

    def test_record_fallback(self):
        """record_fallback should update decision."""
        with AIDecisionContext(auto_log=False) as ctx:
            ctx.record_fallback("timeout")

        assert ctx.decision.used_fallback is True
        assert ctx.decision.fallback_reason == "timeout"

    def test_set_game_phase(self):
        """set_game_phase should update decision."""
        with AIDecisionContext(auto_log=False) as ctx:
            ctx.set_game_phase("placement", pieces_in_hand=10)

        assert ctx.decision.game_phase == "placement"
        assert ctx.decision.pieces_in_hand == 10

    def test_mark_time(self):
        """mark_time should record time breakdowns."""
        with AIDecisionContext(auto_log=False) as ctx:
            time.sleep(0.01)
            ctx.mark_time("after_search")
            time.sleep(0.01)
            ctx.mark_time("after_eval")

        assert "after_search" in ctx.decision.time_breakdown
        assert "after_eval" in ctx.decision.time_breakdown
        # Second mark should be later
        assert ctx.decision.time_breakdown["after_eval"] > ctx.decision.time_breakdown["after_search"]

    def test_exception_handling(self):
        """Context manager should record errors."""
        with pytest.raises(ValueError):
            with AIDecisionContext(auto_log=False) as ctx:
                raise ValueError("Test error")

        assert ctx.decision.error == "Test error"


class TestTrackAIDecision:
    """Tests for track_ai_decision helper."""

    def test_basic_usage(self):
        """track_ai_decision should work as context manager."""
        with track_ai_decision(
            game_id="track_test",
            difficulty=3,
            engine_type="mcts",
            auto_log=False,
        ) as ctx:
            ctx.record_move("test_move", score=0.5)

        assert ctx.decision.game_id == "track_test"
        assert ctx.decision.difficulty == 3
        assert ctx.decision.chosen_move == "test_move"


class TestCreateDecisionLogFromStats:
    """Tests for create_decision_log_from_stats helper."""

    def test_basic_conversion(self):
        """Should convert stats dict to AIDecisionLog."""
        stats = {
            'engine_type': 'mcts',
            'score': 0.75,
            'time_ms': 150.0,
            'simulations': 800,
            'depth': 0,
            'nodes': 5000,
        }

        log = create_decision_log_from_stats(
            move="test_move",
            stats=stats,
            game_id="stats_test",
            difficulty=3,
        )

        assert log.game_id == "stats_test"
        assert log.difficulty == 3
        assert log.engine_type == "mcts"
        assert log.move_score == 0.75
        assert log.time_ms == 150.0
        assert log.simulations == 800
        assert log.chosen_move == "test_move"

    def test_alternate_keys(self):
        """Should handle alternate key names."""
        stats = {
            'type': 'minimax',  # alternate for engine_type
            'value': 0.5,  # alternate for score
            'elapsed_ms': 100.0,  # alternate for time_ms
            'sims': 400,  # alternate for simulations
            'nodes_evaluated': 3000,  # alternate for nodes
            'model': 'v2.1',  # alternate for model_version
        }

        log = create_decision_log_from_stats(move="alt_move", stats=stats)

        assert log.engine_type == "minimax"
        assert log.move_score == 0.5
        assert log.time_ms == 100.0
        assert log.simulations == 400
        assert log.nodes_evaluated == 3000
        assert log.model_version == "v2.1"

    def test_none_move(self):
        """Should handle None move."""
        log = create_decision_log_from_stats(move=None, stats={})
        assert log.chosen_move == ""


class TestLogAIDecision:
    """Tests for log_ai_decision function."""

    def test_logs_without_error(self, caplog):
        """log_ai_decision should log at INFO level normally."""
        import logging
        caplog.set_level(logging.INFO)

        decision = AIDecisionLog(
            engine_type="test",
            chosen_move="move",
            move_score=0.5,
            time_ms=100.0,
        )

        log_ai_decision(decision)

        # Should have logged something
        assert len(caplog.records) > 0

    def test_logs_error_level_on_error(self, caplog):
        """log_ai_decision should log at ERROR level when error present."""
        import logging
        caplog.set_level(logging.ERROR)

        decision = AIDecisionLog(
            engine_type="test",
            chosen_move="move",
            move_score=0.0,
            time_ms=100.0,
            error="Something went wrong",
        )

        log_ai_decision(decision, log_level=logging.ERROR)

        # Should have logged at error level
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) > 0
