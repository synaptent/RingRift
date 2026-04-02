"""Tests for SafeEventEmitterMixin and module-level safe_emit_event.

These tests validate the current delegation contract:
- mixin methods forward to event_emission_helpers
- custom event sources become helper context/source
- async emission preserves return values
- the module remains lazily importable
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.safe_event_emitter import SafeEventEmitterMixin, safe_emit_event


class TestEmitter(SafeEventEmitterMixin):
    """Test class that uses the mixin."""

    _event_source = "TestEmitter"


class CustomSourceEmitter(SafeEventEmitterMixin):
    """Test class with custom source."""

    _event_source = "CustomSource"


@pytest.fixture
def emitter() -> TestEmitter:
    """Create a test emitter instance."""
    return TestEmitter()


@pytest.fixture
def custom_emitter() -> CustomSourceEmitter:
    """Create a custom source emitter instance."""
    return CustomSourceEmitter()


class TestSafeEventEmitterMixin:
    """Tests for SafeEventEmitterMixin."""

    def test_event_source_default(self):
        """Default event source falls back to unknown."""

        class DefaultEmitter(SafeEventEmitterMixin):
            pass

        assert DefaultEmitter()._event_source == "unknown"

    def test_event_source_custom(self, emitter: TestEmitter):
        """Custom event source is exposed on the instance."""
        assert emitter._event_source == "TestEmitter"

    def test_safe_emit_event_delegates_to_helper(self, emitter: TestEmitter):
        """Sync emission delegates to the consolidated helper."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            result = emitter._safe_emit_event("TEST_EVENT", {"key": "value"})

        assert result is True
        mock_emit.assert_called_once_with(
            "TEST_EVENT",
            {"key": "value"},
            log_before=None,
            log_after=None,
            context="TestEmitter",
            source="TestEmitter",
        )

    def test_safe_emit_event_passes_none_payload_through(self, emitter: TestEmitter):
        """None payload is delegated unchanged for helper normalization."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            emitter._safe_emit_event("EMPTY_EVENT", None)

        mock_emit.assert_called_once_with(
            "EMPTY_EVENT",
            None,
            log_before=None,
            log_after=None,
            context="TestEmitter",
            source="TestEmitter",
        )

    def test_safe_emit_event_propagates_false(self, emitter: TestEmitter):
        """False from the helper is returned to the caller."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=False,
        ):
            result = emitter._safe_emit_event("TEST_EVENT")

        assert result is False

    def test_safe_emit_event_uses_custom_source(
        self, custom_emitter: CustomSourceEmitter
    ):
        """Custom emitters pass their own context and source."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            custom_emitter._safe_emit_event("TEST_EVENT")

        mock_emit.assert_called_once_with(
            "TEST_EVENT",
            None,
            log_before=None,
            log_after=None,
            context="CustomSource",
            source="CustomSource",
        )


class TestSafeEventEmitterMixinAsync:
    """Tests for async emission."""

    @pytest.mark.asyncio
    async def test_safe_emit_event_async_success(self, emitter: TestEmitter):
        """Async emission returns the helper result."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event_async",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            result = await emitter._safe_emit_event_async(
                "ASYNC_EVENT",
                {"async": True},
            )

        assert result is True
        mock_emit.assert_awaited_once_with(
            "ASYNC_EVENT",
            {"async": True},
            log_before=None,
            log_after=None,
            context="TestEmitter",
            source="TestEmitter",
        )

    @pytest.mark.asyncio
    async def test_safe_emit_event_async_failure(self, emitter: TestEmitter):
        """Async emission propagates a False helper result."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event_async",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await emitter._safe_emit_event_async("ASYNC_EVENT")

        assert result is False


class TestSafeEmitEventFunction:
    """Tests for the module-level helper."""

    def test_safe_emit_event_delegates_to_helper(self):
        """Module helper forwards args and default source."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            result = safe_emit_event("MODULE_EVENT", {"key": "value"})

        assert result is True
        mock_emit.assert_called_once_with(
            "MODULE_EVENT",
            {"key": "value"},
            log_before=None,
            log_after=None,
            context="module",
            source="module",
        )

    def test_safe_emit_event_custom_source(self):
        """Custom source becomes both source and default context."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            safe_emit_event(
                "MODULE_EVENT",
                {"key": "value"},
                source="my_custom_module",
            )

        mock_emit.assert_called_once_with(
            "MODULE_EVENT",
            {"key": "value"},
            log_before=None,
            log_after=None,
            context="my_custom_module",
            source="my_custom_module",
        )

    def test_safe_emit_event_explicit_context(self):
        """Explicit context overrides the source-derived default."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=True,
        ) as mock_emit:
            safe_emit_event(
                "MODULE_EVENT",
                {"key": "value"},
                source="source_name",
                context="explicit_context",
            )

        mock_emit.assert_called_once_with(
            "MODULE_EVENT",
            {"key": "value"},
            log_before=None,
            log_after=None,
            context="explicit_context",
            source="source_name",
        )

    def test_safe_emit_event_propagates_false(self):
        """Module helper returns False when the consolidated helper fails."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            return_value=False,
        ):
            result = safe_emit_event("MODULE_EVENT")

        assert result is False


class TestSafeEventEmitterIntegration:
    """Lightweight integration coverage."""

    def test_lazy_import_avoids_circular_dependency(self):
        """Creating an emitter does not require eager router imports."""
        from app.coordination.safe_event_emitter import SafeEventEmitterMixin as ImportedMixin

        class LazyEmitter(ImportedMixin):
            _event_source = "LazyEmitter"

        assert LazyEmitter()._event_source == "LazyEmitter"

    def test_sync_helper_stays_usable_without_running_loop(self):
        """The consolidated sync helper still works in a plain sync context."""
        mock_safe_emit = MagicMock(return_value=True)

        with patch(
            "app.coordination.event_router.safe_emit_event",
            mock_safe_emit,
        ):
            result = safe_emit_event("SYNC_EVENT", {"value": 1}, source="sync_source")

        assert result is True
        mock_safe_emit.assert_called_once_with(
            "SYNC_EVENT",
            {"value": 1},
            source="sync_source",
            log_on_failure=False,
        )
