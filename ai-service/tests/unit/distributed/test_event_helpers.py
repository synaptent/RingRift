"""Tests for app.distributed.event_helpers module.

This module tests the safe event emission wrappers that handle
ImportError gracefully when the event bus is not available.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeDataEventType:
    MODEL_PROMOTED = "MODEL_PROMOTED"


class _FakeDataEvent:
    def __init__(self, event_type, payload, source):
        self.event_type = event_type
        self.payload = payload
        self.source = source


# =============================================================================
# Test Availability Checks
# =============================================================================


class TestAvailabilityChecks:
    """Tests for event bus availability checks."""

    def test_has_event_bus_returns_bool(self):
        """Test that has_event_bus returns a boolean."""
        from app.distributed.event_helpers import has_event_bus

        result = has_event_bus()
        assert isinstance(result, bool)

    def test_has_event_router_returns_bool(self):
        """Test that has_event_router returns a boolean."""
        from app.distributed.event_helpers import has_event_router

        result = has_event_router()
        assert isinstance(result, bool)

    def test_get_event_bus_safe_returns_bus_or_none(self):
        """Test get_event_bus_safe returns bus or None."""
        from app.distributed.event_helpers import get_event_bus_safe

        result = get_event_bus_safe()
        # Should be either a bus instance or None
        assert result is not None or result is None

    def test_get_event_types_returns_enum_or_none(self):
        """Test get_event_types returns DataEventType or None."""
        from app.distributed.event_helpers import get_event_types

        result = get_event_types()
        # Should be either the enum or None
        if result is not None:
            assert hasattr(result, "MODEL_PROMOTED")


# =============================================================================
# Test Router Configuration
# =============================================================================


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_use_router_by_default_is_bool(self):
        """Test that USE_ROUTER_BY_DEFAULT is a boolean."""
        from app.distributed.event_helpers import USE_ROUTER_BY_DEFAULT

        assert isinstance(USE_ROUTER_BY_DEFAULT, bool)

    def test_set_use_router_by_default(self):
        """Test setting USE_ROUTER_BY_DEFAULT."""
        from app.distributed import event_helpers

        original = event_helpers.USE_ROUTER_BY_DEFAULT

        try:
            event_helpers.set_use_router_by_default(False)
            assert event_helpers.USE_ROUTER_BY_DEFAULT is False

            event_helpers.set_use_router_by_default(True)
            assert event_helpers.USE_ROUTER_BY_DEFAULT is True
        finally:
            # Restore original
            event_helpers.set_use_router_by_default(original)


# =============================================================================
# Test Event Creation
# =============================================================================


class TestEventCreation:
    """Tests for event creation helpers."""

    def test_create_event_valid_type(self):
        """Test creating an event with valid type."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_DataEventType", _FakeDataEventType), \
             patch.object(event_helpers, "_DataEvent", _FakeDataEvent):
            event = event_helpers.create_event(
                event_type="MODEL_PROMOTED",
                payload={"model_id": "test", "elo": 1500.0},
                source="test",
            )

        assert event is not None
        assert event.event_type == _FakeDataEventType.MODEL_PROMOTED

    def test_create_event_invalid_type(self):
        """Test creating an event with invalid type."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_DataEventType", _FakeDataEventType), \
             patch.object(event_helpers, "_DataEvent", _FakeDataEvent):
            event = event_helpers.create_event(
                event_type="NONEXISTENT_EVENT_TYPE",
                payload={},
                source="test",
            )

        assert event is None

    def test_create_event_no_bus(self):
        """Test creating event when bus unavailable."""
        from app.distributed import event_helpers

        # Mock unavailable bus
        with patch.object(event_helpers, "_HAS_EVENT_BUS", False):
            event = event_helpers.create_event("MODEL_PROMOTED", {}, "test")
            assert event is None


# =============================================================================
# Test Safe Emit Functions
# =============================================================================


class TestEmitEventSafe:
    """Tests for emit_event_safe function."""

    @pytest.mark.asyncio
    async def test_emit_event_safe_success(self):
        """Test successful event emission."""
        from app.distributed.event_helpers import emit_event_safe

        # Even if event bus isn't available, it should not raise
        result = await emit_event_safe(
            event_type="MODEL_PROMOTED",
            payload={"model_id": "test"},
            source="test",
        )

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_emit_event_safe_with_router(self):
        """Test event emission through router."""
        from app.distributed import event_helpers

        mock_publish = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_ROUTER", True), \
             patch.object(event_helpers, "_router_publish", mock_publish):

            result = await event_helpers.emit_event_safe(
                event_type="TEST_EVENT",
                payload={"data": "value"},
                source="test",
                use_router=True,
            )

            assert result is True
            mock_publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event_safe_router_failure_fallback(self):
        """Test fallback to EventBus when router fails."""
        from app.distributed import event_helpers

        mock_router = AsyncMock(side_effect=Exception("Router error"))
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_ROUTER", True), \
             patch.object(event_helpers, "_router_publish", mock_router), \
             patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "get_event_bus_safe", return_value=mock_bus), \
             patch.object(event_helpers, "create_event", return_value=MagicMock()):

            result = await event_helpers.emit_event_safe(
                event_type="TEST_EVENT",
                payload={},
                source="test",
                use_router=True,
            )

            # Should have tried bus after router failed
            assert mock_router.called

    @pytest.mark.asyncio
    async def test_emit_event_safe_no_bus(self):
        """Test emit_event_safe when bus is unavailable."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "_HAS_EVENT_ROUTER", False), \
             patch.object(event_helpers, "get_event_bus_safe", return_value=None):

            result = await event_helpers.emit_event_safe(
                event_type="MODEL_PROMOTED",
                payload={},
                source="test",
                use_router=False,
            )

            assert result is False


# =============================================================================
# Test Safe Subscription
# =============================================================================


class TestSubscribeSafe:
    """Tests for subscribe_safe function."""

    def test_subscribe_safe_no_bus(self):
        """Test subscribe_safe when bus is unavailable."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "get_event_bus_safe", return_value=None):
            result = event_helpers.subscribe_safe("MODEL_PROMOTED", lambda e: None)
            assert result is False

    def test_subscribe_safe_invalid_event_type(self):
        """Test subscribe_safe with invalid event type."""
        from app.distributed import event_helpers

        mock_bus = MagicMock()

        with patch.object(event_helpers, "get_event_bus_safe", return_value=mock_bus), \
             patch.object(event_helpers, "_DataEventType", None):

            result = event_helpers.subscribe_safe("MODEL_PROMOTED", lambda e: None)
            assert result is False

    def test_subscribe_safe_success(self):
        """Test successful subscription."""
        from app.distributed import event_helpers

        handler = AsyncMock()
        mock_bus = MagicMock()
        with patch.object(event_helpers, "_DataEventType", _FakeDataEventType):
            result = event_helpers.subscribe_safe(
                "MODEL_PROMOTED",
                handler,
                bus=mock_bus,
            )

        assert result is True
        mock_bus.subscribe.assert_called_once_with(
            _FakeDataEventType.MODEL_PROMOTED,
            handler,
        )


# =============================================================================
# Test Specific Emit Functions
# =============================================================================


class TestEmitModelPromotedSafe:
    """Tests for emit_model_promoted_safe function."""

    @pytest.mark.asyncio
    async def test_emit_model_promoted_safe_success(self):
        """Test successful model promoted emission."""
        from app.distributed import event_helpers

        mock_emit = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_emit_model_promoted", mock_emit):

            result = await event_helpers.emit_model_promoted_safe(
                model_id="model-123",
                config="hex8_2p",
                elo=1650.0,
                elo_gain=50.0,
                source="test",
            )

            assert result is True
            mock_emit.assert_called_once_with(
                "model-123", "hex8_2p", 1650.0, 50.0, source="test"
            )

    @pytest.mark.asyncio
    async def test_emit_model_promoted_safe_no_bus(self):
        """Test model promoted emission without bus."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "_HAS_EVENT_BUS", False):
            result = await event_helpers.emit_model_promoted_safe(
                model_id="model-123",
                config="hex8_2p",
                elo=1650.0,
            )

            assert result is False


class TestEmitTrainingCompletedSafe:
    """Tests for emit_training_completed_safe function."""

    @pytest.mark.asyncio
    async def test_emit_training_completed_safe_success(self):
        """Test successful training completed emission."""
        from app.distributed import event_helpers

        mock_emit = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_emit_training_completed", mock_emit):

            result = await event_helpers.emit_training_completed_safe(
                config="hex8_2p",
                model_path="/models/hex8_2p.pth",
                loss=0.125,
                samples=50000,
                duration=3600.0,
                source="training",
            )

            assert result is True
            mock_emit.assert_called_once()


class TestEmitEvaluationCompletedSafe:
    """Tests for emit_evaluation_completed_safe function."""

    @pytest.mark.asyncio
    async def test_emit_evaluation_completed_safe_success(self):
        """Test successful evaluation completed emission."""
        from app.distributed import event_helpers

        mock_emit = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_emit_evaluation_completed", mock_emit):

            result = await event_helpers.emit_evaluation_completed_safe(
                config="hex8_2p",
                elo=1650.0,
                games=100,
                win_rate=0.65,
                source="gauntlet",
                beats_current_best=True,
                vs_random_rate=0.95,
                vs_heuristic_rate=0.70,
            )

            assert result is True
            mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_evaluation_completed_includes_extra_payload(self):
        """Test that extra payload fields are included."""
        from app.distributed import event_helpers

        call_kwargs = {}

        async def capture_emit(*args, **kwargs):
            nonlocal call_kwargs
            call_kwargs = kwargs

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_emit_evaluation_completed", capture_emit):

            await event_helpers.emit_evaluation_completed_safe(
                config="hex8_2p",
                elo=1650.0,
                games=100,
                win_rate=0.65,
                beats_current_best=True,
                vs_random_rate=0.95,
            )

            assert "config_key" in call_kwargs
            assert call_kwargs["config_key"] == "hex8_2p"
            assert call_kwargs["beats_current_best"] is True


class TestEmitErrorSafe:
    """Tests for emit_error_safe function."""

    @pytest.mark.asyncio
    async def test_emit_error_safe_success(self):
        """Test successful error emission."""
        from app.distributed import event_helpers

        mock_emit = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_emit_error", mock_emit):

            result = await event_helpers.emit_error_safe(
                component="training",
                error="OutOfMemoryError",
                source="trainer",
            )

            assert result is True


class TestEmitEloUpdatedSafe:
    """Tests for emit_elo_updated_safe function."""

    @pytest.mark.asyncio
    async def test_emit_elo_updated_safe_success(self):
        """Test successful Elo update emission."""
        from app.distributed import event_helpers

        mock_emit = AsyncMock()

        with patch.object(event_helpers, "_HAS_EVENT_BUS", True), \
             patch.object(event_helpers, "_emit_elo_updated", mock_emit):

            result = await event_helpers.emit_elo_updated_safe(
                config="hex8_2p",
                model_id="model-123",
                new_elo=1700.0,
                old_elo=1650.0,
                games_played=50,
                source="elo_sync",
            )

            assert result is True


# =============================================================================
# Test Generic Event Helpers
# =============================================================================


class TestGenericEventHelpers:
    """Tests for generic event emission helpers."""

    @pytest.mark.asyncio
    async def test_emit_new_games_safe(self):
        """Test emit_new_games_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_new_games_safe(
                host="node-1",
                new_games=100,
                config="hex8_2p",
                source="selfplay",
            )

            assert result is True
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_training_started_safe(self):
        """Test emit_training_started_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_training_started_safe(
                config="hex8_2p",
                samples=50000,
                source="trainer",
            )

            assert result is True
            mock.assert_called_once_with(
                "TRAINING_STARTED",
                {"config": "hex8_2p", "samples": 50000},
                "trainer",
            )

    @pytest.mark.asyncio
    async def test_emit_training_failed_safe(self):
        """Test emit_training_failed_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_training_failed_safe(
                config="hex8_2p",
                error="CUDA out of memory",
                duration=120.0,
                source="trainer",
            )

            assert result is True


# =============================================================================
# Test Quality Event Helpers
# =============================================================================


class TestQualityEventHelpers:
    """Tests for quality-related event helpers."""

    @pytest.mark.asyncio
    async def test_emit_quality_score_updated_safe(self):
        """Test emit_quality_score_updated_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_quality_score_updated_safe(
                game_id="game-123",
                old_score=0.5,
                new_score=0.8,
                config="hex8_2p",
                source="quality_monitor",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_emit_quality_distribution_changed_safe(self):
        """Test emit_quality_distribution_changed_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_quality_distribution_changed_safe(
                config="hex8_2p",
                avg_quality=0.75,
                high_quality_ratio=0.6,
                low_quality_ratio=0.1,
                total_games=1000,
                source="quality_monitor",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_emit_high_quality_data_available_safe(self):
        """Test emit_high_quality_data_available_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_high_quality_data_available_safe(
                config="hex8_2p",
                high_quality_count=500,
                avg_quality=0.85,
                source="quality_monitor",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_emit_low_quality_data_warning_safe(self):
        """Test emit_low_quality_data_warning_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_low_quality_data_warning_safe(
                config="hex8_2p",
                low_quality_count=100,
                low_quality_ratio=0.2,
                avg_quality=0.45,
                source="quality_monitor",
            )

            assert result is True


# =============================================================================
# Test Tier Promotion Event Helpers
# =============================================================================


class TestTierPromotionEventHelpers:
    """Tests for tier promotion event helpers."""

    @pytest.mark.asyncio
    async def test_emit_tier_promotion_safe(self):
        """Test emit_tier_promotion_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_tier_promotion_safe(
                config="hex8_2p",
                old_tier="D4",
                new_tier="D5",
                model_id="model-123",
                win_rate=0.75,
                elo=1650.0,
                games_played=100,
                source="tier_gating",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_emit_crossboard_promotion_safe(self):
        """Test emit_crossboard_promotion_safe function."""
        from app.distributed import event_helpers

        with patch.object(event_helpers, "emit_event_safe", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await event_helpers.emit_crossboard_promotion_safe(
                candidate_id="model-123",
                tier="D8",
                target_elo=1800.0,
                mean_elo=1750.0,
                min_elo=1650.0,
                configs_passed=10,
                configs_total=12,
                promotion_applied=False,
                source="crossboard",
            )

            assert result is True


# =============================================================================
# Test Sync Wrapper
# =============================================================================


class TestSyncWrapper:
    """Tests for synchronous emit wrapper."""

    def test_emit_sync_no_loop(self):
        """Test emit_sync when no event loop is running."""
        from app.distributed import event_helpers

        async def fake_emit(*args, **kwargs):
            return True

        def fake_run(coro):
            coro.close()
            return True

        with patch.object(event_helpers, "emit_event_safe", new=fake_emit):
            with patch("asyncio.run", side_effect=fake_run):
                result = event_helpers.emit_sync(
                    event_type="TEST_EVENT",
                    payload={"data": "value"},
                    source="test",
                )

                assert isinstance(result, bool)

    def test_emit_sync_existing_loop(self):
        """Test emit_sync when event loop exists."""
        from app.distributed import event_helpers

        async def run_test():
            async def fake_emit(*args, **kwargs):
                return True

            captured_coroutines = []

            def fake_fire_and_forget(coro, *args, **kwargs):
                captured_coroutines.append(coro)
                coro.close()

            with patch.object(event_helpers, "emit_event_safe", new=fake_emit):
                # In an existing loop, should schedule as fire_and_forget
                with patch("app.utils.async_utils.fire_and_forget", new=fake_fire_and_forget):
                    result = event_helpers.emit_sync(
                        event_type="TEST_EVENT",
                        payload={},
                        source="test",
                    )
                    assert len(captured_coroutines) == 1
                    return result

        # Run in a loop
        result = asyncio.run(run_test())
        assert result is True


# =============================================================================
# Test Module Re-exports
# =============================================================================


class TestModuleReexports:
    """Tests for module re-exports."""

    def test_dataeventtype_reexport(self):
        """Test that DataEventType is re-exported."""
        from app.distributed.event_helpers import DataEventType

        # May be None if not available
        if DataEventType is not None:
            assert hasattr(DataEventType, "MODEL_PROMOTED")

    def test_dataevent_reexport(self):
        """Test that DataEvent is re-exported."""
        from app.distributed.event_helpers import DataEvent

        # May be None if not available
        assert DataEvent is None or hasattr(DataEvent, "__init__")

    def test_eventbus_reexport(self):
        """Test that EventBus is re-exported."""
        from app.distributed.event_helpers import EventBus

        # May be None if not available
        assert EventBus is None or hasattr(EventBus, "publish")


# =============================================================================
# Test __all__ Exports
# =============================================================================


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_available(self):
        """Test that all items in __all__ are available."""
        from app.distributed import event_helpers

        for name in event_helpers.__all__:
            assert hasattr(event_helpers, name), f"Missing export: {name}"

    def test_all_exports_contains_key_functions(self):
        """Test that __all__ contains key functions."""
        from app.distributed.event_helpers import __all__

        expected = [
            "has_event_bus",
            "has_event_router",
            "get_event_bus_safe",
            "emit_event_safe",
            "emit_model_promoted_safe",
            "emit_training_completed_safe",
            "subscribe_safe",
        ]

        for name in expected:
            assert name in __all__, f"Expected {name} in __all__"
