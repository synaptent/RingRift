"""Unit tests for event_handler_decorator module.

Tests the unified event handler decorator for standardized event handling
patterns across the coordination layer.

December 2025: Created as part of event handler pattern standardization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.coordination.event_handler_decorator import (
    STANDARD_FIELDS,
    EventContext,
    FieldSpec,
    HandlerMetrics,
    event_handler,
    extract_with_context,
    get_all_handler_metrics,
    get_handler_metrics,
    normalize_event_payload,
    reset_handler_metrics,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset handler metrics before each test."""
    reset_handler_metrics()
    yield
    reset_handler_metrics()


@pytest.fixture
def sample_event():
    """Sample event with .payload attribute."""

    class Event:
        def __init__(self, payload: dict):
            self.payload = payload
            self.event_type = "TEST_EVENT"
            self.source = "test_source"

    return Event(
        {
            "config_key": "hex8_2p",
            "model_path": "/path/to/model.pth",
            "elo": 1500.5,
            "games_played": 100,
            "board_type": "hex8",
            "num_players": 2,
        }
    )


@pytest.fixture
def dict_event():
    """Sample event as plain dict."""
    return {
        "config_key": "square8_4p",
        "model_path": "/path/to/model2.pth",
        "elo": 1600,
        "games_played": 200,
    }


# =============================================================================
# EventContext Tests
# =============================================================================


class TestEventContext:
    """Tests for EventContext class."""

    def test_basic_creation(self):
        """Should create context with payload."""
        ctx = EventContext(
            raw_event={},
            payload={"config_key": "hex8_2p"},
        )
        assert ctx.payload == {"config_key": "hex8_2p"}
        assert ctx.event_type == ""
        assert ctx.source == ""

    def test_get_method(self):
        """Should get field with default."""
        ctx = EventContext(
            raw_event={},
            payload={"field": "value"},
        )
        assert ctx.get("field") == "value"
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_get_int(self):
        """Should coerce to int."""
        ctx = EventContext(
            raw_event={},
            payload={"count": "42", "invalid": "abc", "float": 3.7},
        )
        assert ctx.get_int("count") == 42
        assert ctx.get_int("invalid") == 0
        assert ctx.get_int("float") == 3
        assert ctx.get_int("missing", 10) == 10

    def test_get_float(self):
        """Should coerce to float."""
        ctx = EventContext(
            raw_event={},
            payload={"rate": "0.75", "invalid": "abc"},
        )
        assert ctx.get_float("rate") == 0.75
        assert ctx.get_float("invalid") == 0.0
        assert ctx.get_float("missing", 1.5) == 1.5

    def test_get_bool(self):
        """Should coerce to bool."""
        ctx = EventContext(
            raw_event={},
            payload={
                "true_str": "true",
                "false_str": "false",
                "one": "1",
                "yes": "yes",
                "actual_bool": True,
            },
        )
        assert ctx.get_bool("true_str") is True
        assert ctx.get_bool("false_str") is False
        assert ctx.get_bool("one") is True
        assert ctx.get_bool("yes") is True
        assert ctx.get_bool("actual_bool") is True
        assert ctx.get_bool("missing") is False
        assert ctx.get_bool("missing", True) is True

    def test_get_list(self):
        """Should get list field."""
        ctx = EventContext(
            raw_event={},
            payload={"items": [1, 2, 3], "not_list": "string"},
        )
        assert ctx.get_list("items") == [1, 2, 3]
        assert ctx.get_list("not_list") == []
        assert ctx.get_list("missing") == []
        assert ctx.get_list("missing", [1]) == [1]

    def test_get_dict(self):
        """Should get dict field."""
        ctx = EventContext(
            raw_event={},
            payload={"data": {"a": 1}, "not_dict": "string"},
        )
        assert ctx.get_dict("data") == {"a": 1}
        assert ctx.get_dict("not_dict") == {}
        assert ctx.get_dict("missing") == {}
        assert ctx.get_dict("missing", {"x": 1}) == {"x": 1}

    def test_config_key_property(self):
        """Should extract config_key."""
        ctx = EventContext(
            raw_event={},
            payload={"config_key": "hex8_2p"},
        )
        assert ctx.config_key == "hex8_2p"

    def test_config_key_from_alias(self):
        """Should extract config_key from 'config' alias."""
        ctx = EventContext(
            raw_event={},
            payload={"config": "square8_4p"},
        )
        assert ctx.config_key == "square8_4p"

    def test_board_type_property(self):
        """Should extract board_type."""
        ctx = EventContext(
            raw_event={},
            payload={"board_type": "hex8"},
        )
        assert ctx.board_type == "hex8"

    def test_board_type_from_config_key(self):
        """Should parse board_type from config_key."""
        ctx = EventContext(
            raw_event={},
            payload={"config_key": "square19_3p"},
        )
        assert ctx.board_type == "square19"

    def test_num_players_property(self):
        """Should extract num_players."""
        ctx = EventContext(
            raw_event={},
            payload={"num_players": 4},
        )
        assert ctx.num_players == 4

    def test_num_players_from_config_key(self):
        """Should parse num_players from config_key."""
        ctx = EventContext(
            raw_event={},
            payload={"config_key": "hexagonal_4p"},
        )
        assert ctx.num_players == 4

    def test_model_path_property(self):
        """Should extract model_path."""
        ctx = EventContext(
            raw_event={},
            payload={"model_path": "/path/to/model.pth"},
        )
        assert ctx.model_path == "/path/to/model.pth"

    def test_model_path_from_alias(self):
        """Should extract model_path from checkpoint_path alias."""
        ctx = EventContext(
            raw_event={},
            payload={"checkpoint_path": "/path/to/checkpoint.pth"},
        )
        assert ctx.model_path == "/path/to/checkpoint.pth"


# =============================================================================
# FieldSpec Tests
# =============================================================================


class TestFieldSpec:
    """Tests for FieldSpec class."""

    def test_basic_extraction(self):
        """Should extract field by name."""
        spec = FieldSpec(name="field")
        value, error = spec.extract({"field": "value"})
        assert value == "value"
        assert error is None

    def test_extraction_with_alias(self):
        """Should try aliases if primary name missing."""
        spec = FieldSpec(name="field", aliases=("alias1", "alias2"))
        value, error = spec.extract({"alias2": "found"})
        assert value == "found"
        assert error is None

    def test_required_field_missing(self):
        """Should return error for missing required field."""
        spec = FieldSpec(name="required_field", required=True)
        value, error = spec.extract({})
        assert value is None
        assert error is not None
        assert "required_field" in error
        assert "not found" in error.lower()

    def test_optional_field_with_default(self):
        """Should return default for missing optional field."""
        spec = FieldSpec(name="optional", default="default_value")
        value, error = spec.extract({})
        assert value == "default_value"
        assert error is None

    def test_type_coercion_int(self):
        """Should coerce to int."""
        spec = FieldSpec(name="count", type_coerce=int)
        value, error = spec.extract({"count": "42"})
        assert value == 42
        assert error is None

    def test_type_coercion_float(self):
        """Should coerce to float."""
        spec = FieldSpec(name="rate", type_coerce=float)
        value, error = spec.extract({"rate": "3.14"})
        assert value == pytest.approx(3.14)
        assert error is None

    def test_type_coercion_bool(self):
        """Should coerce to bool."""
        spec = FieldSpec(name="flag", type_coerce=bool)
        value, error = spec.extract({"flag": "true"})
        assert value is True
        assert error is None

    def test_type_coercion_failure(self):
        """Should return error on coercion failure."""
        spec = FieldSpec(name="count", type_coerce=int, default=0)
        value, error = spec.extract({"count": "not_a_number"})
        assert value == 0  # Default returned
        assert error is not None
        assert "coercion failed" in error.lower()


class TestStandardFields:
    """Tests for STANDARD_FIELDS definitions."""

    def test_config_key_spec(self):
        """Should have config_key with config alias."""
        spec = STANDARD_FIELDS["config_key"]
        assert spec.name == "config_key"
        assert "config" in spec.aliases
        assert spec.type_coerce is str

    def test_model_path_spec(self):
        """Should have model_path with aliases."""
        spec = STANDARD_FIELDS["model_path"]
        assert spec.name == "model_path"
        assert "checkpoint_path" in spec.aliases
        assert "path" in spec.aliases

    def test_elo_spec(self):
        """Should have elo with float coercion."""
        spec = STANDARD_FIELDS["elo"]
        assert spec.type_coerce is float

    def test_games_played_spec(self):
        """Should have games_played with multiple aliases."""
        spec = STANDARD_FIELDS["games_played"]
        assert "games" in spec.aliases
        assert "game_count" in spec.aliases
        assert spec.type_coerce is int


# =============================================================================
# Payload Normalization Tests
# =============================================================================


class TestNormalizeEventPayload:
    """Tests for normalize_event_payload function."""

    def test_dict_event(self):
        """Should pass through dict events."""
        event = {"config_key": "hex8_2p"}
        payload = normalize_event_payload(event)
        assert payload == {"config_key": "hex8_2p"}

    def test_event_with_payload_attr(self):
        """Should extract .payload attribute."""

        class Event:
            payload = {"config_key": "hex8_2p"}

        payload = normalize_event_payload(Event())
        assert payload == {"config_key": "hex8_2p"}

    def test_event_with_metadata_attr(self):
        """Should fall back to .metadata attribute."""

        class Event:
            metadata = {"config_key": "hex8_2p"}

        payload = normalize_event_payload(Event())
        assert payload == {"config_key": "hex8_2p"}

    def test_none_event(self):
        """Should return empty dict for None."""
        payload = normalize_event_payload(None)
        assert payload == {}

    def test_invalid_event(self):
        """Should return empty dict for invalid types."""
        payload = normalize_event_payload("not_an_event")
        assert payload == {}

    def test_payload_precedence(self):
        """Should prefer .payload over .metadata."""

        class Event:
            payload = {"source": "payload"}
            metadata = {"source": "metadata"}

        payload = normalize_event_payload(Event())
        assert payload["source"] == "payload"


# =============================================================================
# extract_with_context Tests
# =============================================================================


class TestExtractWithContext:
    """Tests for extract_with_context function."""

    def test_creates_context(self, sample_event):
        """Should create EventContext from event."""
        ctx = extract_with_context(sample_event)
        assert isinstance(ctx, EventContext)
        assert ctx.config_key == "hex8_2p"
        assert ctx.event_type == "TEST_EVENT"
        assert ctx.source == "test_source"

    def test_with_dict_event(self, dict_event):
        """Should work with dict events."""
        ctx = extract_with_context(dict_event)
        assert ctx.config_key == "square8_4p"
        assert ctx.get("elo") == 1600


# =============================================================================
# event_handler Decorator Tests - Sync Handlers
# =============================================================================


class TestEventHandlerSync:
    """Tests for event_handler decorator with sync handlers."""

    def test_basic_handler(self, sample_event):
        """Should call handler with context and extracted fields."""
        received = {}

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                received["ctx"] = ctx
                received["config_key"] = config_key

        handler = Handler()
        handler._on_event(sample_event)

        assert received["config_key"] == "hex8_2p"
        assert isinstance(received["ctx"], EventContext)

    def test_required_and_optional(self, sample_event):
        """Should extract both required and optional fields."""
        received = {}

        class Handler:
            @event_handler(required=["config_key"], optional=["elo", "model_path"])
            def _on_event(
                self, ctx: EventContext, config_key: str, elo: float, model_path: str
            ) -> None:
                received["config_key"] = config_key
                received["elo"] = elo
                received["model_path"] = model_path

        handler = Handler()
        handler._on_event(sample_event)

        assert received["config_key"] == "hex8_2p"
        assert received["elo"] == pytest.approx(1500.5)
        assert received["model_path"] == "/path/to/model.pth"

    def test_missing_required_field(self):
        """Should not call handler if required field missing."""
        called = False

        class Handler:
            @event_handler(required=["missing_field"])
            def _on_event(self, ctx: EventContext, missing_field: str) -> None:
                nonlocal called
                called = True

        handler = Handler()
        handler._on_event({"config_key": "hex8_2p"})

        assert not called

    def test_optional_field_default(self):
        """Should use default for missing optional field."""
        received = {}

        class Handler:
            @event_handler(optional=["missing_optional"])
            def _on_event(self, ctx: EventContext, missing_optional: str) -> None:
                received["missing_optional"] = missing_optional

        handler = Handler()
        handler._on_event({"config_key": "hex8_2p"})

        assert received["missing_optional"] is None  # Default from FieldSpec

    def test_dict_event(self, dict_event):
        """Should work with dict events."""
        received = {}

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                received["config_key"] = config_key

        handler = Handler()
        handler._on_event(dict_event)

        assert received["config_key"] == "square8_4p"

    def test_return_value(self, sample_event):
        """Should pass through handler return value."""

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> str:
                return f"processed_{config_key}"

        handler = Handler()
        result = handler._on_event(sample_event)

        assert result == "processed_hex8_2p"


# =============================================================================
# event_handler Decorator Tests - Async Handlers
# =============================================================================


class TestEventHandlerAsync:
    """Tests for event_handler decorator with async handlers."""

    @pytest.mark.asyncio
    async def test_async_handler(self, sample_event):
        """Should work with async handlers."""
        received = {}

        class Handler:
            @event_handler(required=["config_key"])
            async def _on_event(self, ctx: EventContext, config_key: str) -> None:
                received["config_key"] = config_key

        handler = Handler()
        await handler._on_event(sample_event)

        assert received["config_key"] == "hex8_2p"

    @pytest.mark.asyncio
    async def test_async_return_value(self, sample_event):
        """Should pass through async handler return value."""

        class Handler:
            @event_handler(required=["config_key"])
            async def _on_event(self, ctx: EventContext, config_key: str) -> str:
                return f"async_{config_key}"

        handler = Handler()
        result = await handler._on_event(sample_event)

        assert result == "async_hex8_2p"


# =============================================================================
# Metrics Tests
# =============================================================================


class TestHandlerMetrics:
    """Tests for handler metrics tracking."""

    def test_metrics_tracked(self, sample_event):
        """Should track handler calls."""

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                pass

        handler = Handler()
        handler._on_event(sample_event)
        handler._on_event(sample_event)

        metrics = get_all_handler_metrics()
        assert len(metrics) == 1

        # Find the handler
        handler_metrics = list(metrics.values())[0]
        assert handler_metrics.calls == 2
        assert handler_metrics.errors == 0

    def test_error_tracking(self):
        """Should track handler errors."""

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                raise ValueError("Test error")

        handler = Handler()
        handler._on_event({"config_key": "hex8_2p"})

        metrics = list(get_all_handler_metrics().values())[0]
        assert metrics.errors == 1

    def test_duration_tracking(self, sample_event):
        """Should track handler duration."""
        import time

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                time.sleep(0.01)  # 10ms

        handler = Handler()
        handler._on_event(sample_event)

        metrics = list(get_all_handler_metrics().values())[0]
        assert metrics.total_duration_ms >= 10  # At least 10ms
        assert metrics.avg_duration_ms >= 10

    def test_metrics_disabled(self, sample_event):
        """Should not track metrics when disabled."""

        class Handler:
            @event_handler(required=["config_key"], track_metrics=False)
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                pass

        handler = Handler()
        handler._on_event(sample_event)

        assert len(get_all_handler_metrics()) == 0

    def test_reset_metrics(self, sample_event):
        """Should reset all metrics."""

        class Handler:
            @event_handler(required=["config_key"])
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                pass

        handler = Handler()
        handler._on_event(sample_event)
        assert len(get_all_handler_metrics()) == 1

        reset_handler_metrics()
        assert len(get_all_handler_metrics()) == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling behavior."""

    def test_error_logged(self, sample_event, caplog):
        """Should log errors by default."""

        class Handler:
            @event_handler(required=["config_key"], on_error="log")
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                raise ValueError("Test error")

        handler = Handler()
        result = handler._on_event(sample_event)

        assert result is None
        assert "Test error" in caplog.text

    def test_error_ignored(self, sample_event, caplog):
        """Should ignore errors when configured."""

        class Handler:
            @event_handler(required=["config_key"], on_error="ignore")
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                raise ValueError("Test error")

        handler = Handler()
        result = handler._on_event(sample_event)

        assert result is None
        # Error should not be logged
        assert "Test error" not in caplog.text

    def test_error_raised(self, sample_event):
        """Should re-raise errors when configured."""

        class Handler:
            @event_handler(required=["config_key"], on_error="raise")
            def _on_event(self, ctx: EventContext, config_key: str) -> None:
                raise ValueError("Test error")

        handler = Handler()
        with pytest.raises(ValueError, match="Test error"):
            handler._on_event(sample_event)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_payload(self):
        """Should handle empty payload."""
        ctx = extract_with_context({})
        assert ctx.config_key == ""
        assert ctx.board_type == ""
        assert ctx.num_players == 0

    def test_none_values_in_payload(self):
        """Should handle None values in payload."""
        ctx = extract_with_context(
            {"config_key": None, "model_path": None, "elo": None}
        )
        assert ctx.config_key == ""
        assert ctx.model_path is None

    def test_wrong_type_in_payload(self):
        """Should handle wrong types in payload."""
        ctx = extract_with_context({"num_players": "not_a_number", "elo": "bad"})
        assert ctx.num_players == 0  # Falls back to parsing from config_key
        assert ctx.get_float("elo") == 0.0

    def test_handler_with_no_fields(self, sample_event):
        """Should work with no required/optional fields."""

        class Handler:
            @event_handler()
            def _on_event(self, ctx: EventContext) -> str:
                return ctx.config_key

        handler = Handler()
        result = handler._on_event(sample_event)
        assert result == "hex8_2p"

    def test_standard_field_type_coercion(self, sample_event):
        """Should apply type coercion for standard fields."""
        received = {}

        class Handler:
            @event_handler(required=["elo", "games_played"])
            def _on_event(
                self, ctx: EventContext, elo: float, games_played: int
            ) -> None:
                received["elo"] = elo
                received["games_played"] = games_played

        # Use string values to test coercion
        handler = Handler()
        handler._on_event({"elo": "1500.5", "games_played": "100"})

        assert received["elo"] == pytest.approx(1500.5)
        assert received["games_played"] == 100


# =============================================================================
# Integration with Existing Code
# =============================================================================


class TestIntegrationWithEventUtils:
    """Tests for integration with event_utils module."""

    def test_context_uses_config_key_parsing(self):
        """Should parse config_key using ConfigKey class."""
        ctx = extract_with_context({"config_key": "hexagonal_4p"})
        assert ctx.board_type == "hexagonal"
        assert ctx.num_players == 4

    def test_context_handles_invalid_config_key(self):
        """Should handle invalid config_key gracefully."""
        ctx = extract_with_context({"config_key": "invalid"})
        # Invalid config_key should return empty values
        assert ctx.board_type == ""
        assert ctx.num_players == 0


# =============================================================================
# HandlerMetrics Tests
# =============================================================================


class TestHandlerMetricsClass:
    """Tests for HandlerMetrics dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        metrics = HandlerMetrics()
        assert metrics.calls == 0
        assert metrics.errors == 0
        assert metrics.total_duration_ms == 0.0
        assert metrics.last_call_time == 0.0

    def test_avg_duration_no_calls(self):
        """Should return 0 for avg_duration with no calls."""
        metrics = HandlerMetrics()
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration_with_calls(self):
        """Should calculate correct average."""
        metrics = HandlerMetrics(calls=4, total_duration_ms=100.0)
        assert metrics.avg_duration_ms == 25.0
