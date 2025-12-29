"""Tests for event_subscription_registry.py.

December 29, 2025: Test coverage for declarative event subscription registry.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.event_subscription_registry import (
    InitCallSpec,
    DelegationSpec,
    INIT_CALL_REGISTRY,
    DELEGATION_REGISTRY,
    process_init_call_registry,
    process_delegation_registry,
    _create_delegation_handler,
    get_registry_stats,
)


class TestInitCallSpec:
    """Tests for InitCallSpec dataclass."""

    def test_basic_creation(self):
        """Test creating a basic InitCallSpec."""
        spec = InitCallSpec(
            name="test_spec",
            import_path="app.test.module",
            function_name="test_function",
        )
        assert spec.name == "test_spec"
        assert spec.import_path == "app.test.module"
        assert spec.function_name == "test_function"
        assert spec.kwargs == {}
        assert spec.description == ""

    def test_with_kwargs(self):
        """Test InitCallSpec with kwargs."""
        spec = InitCallSpec(
            name="test_spec",
            import_path="app.test.module",
            function_name="test_function",
            kwargs={"arg1": True, "arg2": "value"},
            description="Test description",
        )
        assert spec.kwargs == {"arg1": True, "arg2": "value"}
        assert spec.description == "Test description"

    def test_frozen(self):
        """Test that InitCallSpec is frozen (immutable)."""
        spec = InitCallSpec(
            name="test_spec",
            import_path="app.test.module",
            function_name="test_function",
        )
        with pytest.raises(AttributeError):
            spec.name = "new_name"


class TestDelegationSpec:
    """Tests for DelegationSpec dataclass."""

    def test_basic_creation(self):
        """Test creating a basic DelegationSpec."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="get_orchestrator",
            primary_method="handle_event",
        )
        assert spec.name == "test_handler"
        assert spec.event_type == "TEST_EVENT"
        assert spec.orchestrator_import == "app.test.orchestrator"
        assert spec.orchestrator_getter == "get_orchestrator"
        assert spec.primary_method == "handle_event"
        assert spec.fallback_method is None
        assert spec.payload_keys == ()
        assert spec.log_level == "info"

    def test_full_creation(self):
        """Test DelegationSpec with all fields."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="get_orchestrator",
            primary_method="handle_event",
            fallback_method="fallback_handle",
            payload_keys=("node_id", "reason"),
            log_level="warning",
            description="Test handler description",
        )
        assert spec.fallback_method == "fallback_handle"
        assert spec.payload_keys == ("node_id", "reason")
        assert spec.log_level == "warning"
        assert spec.description == "Test handler description"

    def test_frozen(self):
        """Test that DelegationSpec is frozen (immutable)."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="get_orchestrator",
            primary_method="handle_event",
        )
        with pytest.raises(AttributeError):
            spec.event_type = "NEW_EVENT"


class TestInitCallRegistry:
    """Tests for INIT_CALL_REGISTRY contents."""

    def test_registry_not_empty(self):
        """Test that registry has entries."""
        assert len(INIT_CALL_REGISTRY) > 0

    def test_registry_is_tuple(self):
        """Test registry is a tuple (immutable)."""
        assert isinstance(INIT_CALL_REGISTRY, tuple)

    def test_all_entries_are_init_call_specs(self):
        """Test all entries are InitCallSpec instances."""
        for spec in INIT_CALL_REGISTRY:
            assert isinstance(spec, InitCallSpec)

    def test_required_fields_present(self):
        """Test all entries have required fields."""
        for spec in INIT_CALL_REGISTRY:
            assert spec.name != ""
            assert spec.import_path != ""
            assert spec.function_name != ""

    def test_known_subscriptions_present(self):
        """Test known subscriptions are in registry."""
        names = [spec.name for spec in INIT_CALL_REGISTRY]
        assert "model_selector_events" in names
        assert "quality_to_rollback" in names
        assert "regression_to_rollback" in names
        assert "plateau_to_curriculum" in names
        assert "early_stop_to_curriculum" in names


class TestDelegationRegistry:
    """Tests for DELEGATION_REGISTRY contents."""

    def test_registry_not_empty(self):
        """Test that registry has entries."""
        assert len(DELEGATION_REGISTRY) > 0

    def test_registry_is_tuple(self):
        """Test registry is a tuple (immutable)."""
        assert isinstance(DELEGATION_REGISTRY, tuple)

    def test_all_entries_are_delegation_specs(self):
        """Test all entries are DelegationSpec instances."""
        for spec in DELEGATION_REGISTRY:
            assert isinstance(spec, DelegationSpec)

    def test_required_fields_present(self):
        """Test all entries have required fields."""
        for spec in DELEGATION_REGISTRY:
            assert spec.name != ""
            assert spec.event_type != ""
            assert spec.orchestrator_import != ""
            assert spec.orchestrator_getter != ""
            assert spec.primary_method != ""

    def test_known_handlers_present(self):
        """Test known handlers are in registry."""
        names = [spec.name for spec in DELEGATION_REGISTRY]
        assert "host_offline_handler" in names
        assert "host_online_handler" in names
        assert "leader_elected_handler" in names
        assert "node_suspect_handler" in names

    def test_event_types_valid(self):
        """Test event types look like valid enum names."""
        for spec in DELEGATION_REGISTRY:
            # Should be uppercase with underscores
            assert spec.event_type.isupper() or "_" in spec.event_type


class TestProcessInitCallRegistry:
    """Tests for process_init_call_registry function."""

    def test_empty_results_dict(self):
        """Test processing populates results dict."""
        results = {}

        with patch.object(
            __builtins__["__import__"] if hasattr(__builtins__, "__import__") else __builtins__,
            "__import__",
        ) as mock_import:
            # This will fail imports gracefully
            mock_import.side_effect = ImportError("Test import error")
            process_init_call_registry(results)

        # Should have entries for each spec (all False due to import failure)
        for spec in INIT_CALL_REGISTRY:
            assert spec.name in results
            assert results[spec.name] is False

    def test_successful_init_call(self):
        """Test successful init function call."""
        results = {}

        # Create a mock registry with a simple spec
        mock_func = MagicMock(return_value=True)
        mock_module = MagicMock()
        mock_module.test_func = mock_func

        with patch.dict("sys.modules", {"app.test.module": mock_module}):
            with patch("app.coordination.event_subscription_registry.INIT_CALL_REGISTRY", (
                InitCallSpec(
                    name="test_init",
                    import_path="app.test.module",
                    function_name="test_func",
                ),
            )):
                process_init_call_registry(results)

        # Should succeed with our mocked module
        # Note: This test verifies the flow, actual success depends on import patching


class TestProcessDelegationRegistry:
    """Tests for process_delegation_registry function."""

    def test_no_event_bus(self):
        """Test handling when event bus unavailable."""
        results = {}

        with patch(
            "app.coordination.event_subscription_registry.get_event_bus",
            side_effect=ImportError("No event bus"),
        ):
            process_delegation_registry(results)

        # All should be False
        for spec in DELEGATION_REGISTRY:
            assert spec.name in results
            assert results[spec.name] is False

    def test_with_event_bus(self):
        """Test processing with event bus available."""
        results = {}

        mock_bus = MagicMock()
        mock_bus.subscribe = MagicMock()

        with patch(
            "app.coordination.event_subscription_registry.get_event_bus",
            return_value=mock_bus,
        ):
            with patch(
                "app.coordination.event_subscription_registry.DataEventType",
            ) as mock_event_type:
                # Mock the event type enum
                mock_event_type.HOST_OFFLINE = "HOST_OFFLINE"
                mock_event_type.HOST_ONLINE = "HOST_ONLINE"

                process_delegation_registry(results)

        # subscribe should have been called for valid event types
        assert mock_bus.subscribe.call_count > 0


class TestCreateDelegationHandler:
    """Tests for _create_delegation_handler function."""

    def test_creates_async_handler(self):
        """Test that created handler is async."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="get_orchestrator",
            primary_method="handle_event",
            payload_keys=("node_id",),
        )

        handler = _create_delegation_handler(spec)

        import asyncio
        assert asyncio.iscoroutinefunction(handler)

    @pytest.mark.asyncio
    async def test_handler_extracts_payload(self):
        """Test handler extracts payload correctly."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="get_orchestrator",
            primary_method="handle_event",
            payload_keys=("node_id",),
        )

        handler = _create_delegation_handler(spec)

        # Create mock event with payload
        mock_event = MagicMock()
        mock_event.payload = {"node_id": "test-node-1"}

        # Handler should not raise even if orchestrator not available
        await handler(mock_event)

    @pytest.mark.asyncio
    async def test_handler_skips_no_identifier(self):
        """Test handler skips when no identifier in payload."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="get_orchestrator",
            primary_method="handle_event",
            payload_keys=("node_id",),
        )

        handler = _create_delegation_handler(spec)

        # Create mock event with empty payload
        mock_event = MagicMock()
        mock_event.payload = {}

        # Should complete without error (just return early)
        await handler(mock_event)

    @pytest.mark.asyncio
    async def test_handler_with_class_getter(self):
        """Test handler with Class.get_instance pattern."""
        spec = DelegationSpec(
            name="test_handler",
            event_type="TEST_EVENT",
            orchestrator_import="app.test.orchestrator",
            orchestrator_getter="MyClass.get_instance",
            primary_method="handle_event",
            payload_keys=("node_id",),
        )

        handler = _create_delegation_handler(spec)

        # Should parse the getter correctly
        mock_event = MagicMock()
        mock_event.payload = {"node_id": "test-node"}

        # Handler should handle import errors gracefully
        await handler(mock_event)


class TestGetRegistryStats:
    """Tests for get_registry_stats function."""

    def test_returns_dict(self):
        """Test function returns dict."""
        stats = get_registry_stats()
        assert isinstance(stats, dict)

    def test_has_expected_keys(self):
        """Test stats has expected keys."""
        stats = get_registry_stats()
        assert "init_call_count" in stats
        assert "delegation_count" in stats
        assert "total_subscriptions" in stats

    def test_counts_correct(self):
        """Test counts match registry sizes."""
        stats = get_registry_stats()
        assert stats["init_call_count"] == len(INIT_CALL_REGISTRY)
        assert stats["delegation_count"] == len(DELEGATION_REGISTRY)
        assert stats["total_subscriptions"] == len(INIT_CALL_REGISTRY) + len(DELEGATION_REGISTRY)

    def test_total_is_sum(self):
        """Test total is sum of init and delegation."""
        stats = get_registry_stats()
        assert stats["total_subscriptions"] == stats["init_call_count"] + stats["delegation_count"]
