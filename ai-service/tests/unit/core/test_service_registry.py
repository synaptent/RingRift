"""Tests for app.core.service_registry.

January 2026: Phase 4.5 testing.
"""

import threading
import pytest

from app.core.service_registry import (
    ServiceRegistry,
    ServiceEntry,
    ServiceName,
    get_registry,
    register_service,
    get_service,
)


class TestServiceRegistry:
    """Tests for the ServiceRegistry class."""

    def setup_method(self):
        """Reset registry before each test."""
        ServiceRegistry.reset()

    def teardown_method(self):
        """Reset registry after each test."""
        ServiceRegistry.reset()

    def test_get_instance_returns_singleton(self):
        """get_instance should return the same instance."""
        instance1 = ServiceRegistry.get_instance()
        instance2 = ServiceRegistry.get_instance()
        assert instance1 is instance2

    def test_reset_clears_registry(self):
        """reset should clear all services."""
        registry = ServiceRegistry.get_instance()
        registry.register("test_service", instance="test_value")
        assert registry.get("test_service") == "test_value"

        ServiceRegistry.reset()
        registry = ServiceRegistry.get_instance()
        assert registry.get("test_service") is None

    def test_register_with_instance(self):
        """register with instance should store immediately."""
        registry = ServiceRegistry.get_instance()
        registry.register("my_service", instance="my_instance")

        assert registry.get("my_service") == "my_instance"
        assert registry.is_registered("my_service")

    def test_register_with_factory(self):
        """register with factory should lazily initialize."""
        call_count = [0]

        def factory():
            call_count[0] += 1
            return f"instance_{call_count[0]}"

        registry = ServiceRegistry.get_instance()
        registry.register("lazy_service", factory=factory)

        # Factory not called yet
        assert call_count[0] == 0

        # First get calls factory
        result = registry.get("lazy_service")
        assert result == "instance_1"
        assert call_count[0] == 1

        # Second get returns cached instance
        result = registry.get("lazy_service")
        assert result == "instance_1"
        assert call_count[0] == 1  # Still 1, not called again

    def test_register_requires_instance_or_factory(self):
        """register should raise if neither instance nor factory provided."""
        registry = ServiceRegistry.get_instance()

        with pytest.raises(ValueError, match="Must provide instance or factory"):
            registry.register("bad_service")

    def test_get_nonexistent_returns_none(self):
        """get for non-existent service should return None."""
        registry = ServiceRegistry.get_instance()
        assert registry.get("nonexistent") is None

    def test_get_or_create(self):
        """get_or_create should create if not registered."""
        registry = ServiceRegistry.get_instance()

        result = registry.get_or_create("new_service", lambda: "created")
        assert result == "created"

        # Subsequent calls return same instance
        result = registry.get_or_create("new_service", lambda: "different")
        assert result == "created"

    def test_list_services(self):
        """list_services should return all registered names."""
        registry = ServiceRegistry.get_instance()
        registry.register("service1", instance="a")
        registry.register("service2", instance="b")
        registry.register("service3", instance="c")

        services = registry.list_services()
        assert set(services) == {"service1", "service2", "service3"}

    def test_initialization_callback(self):
        """on_service_initialized should trigger callback."""
        registry = ServiceRegistry.get_instance()
        callbacks = []

        registry.on_service_initialized(
            lambda name, instance: callbacks.append((name, instance))
        )

        # Register with instance - callback fires immediately
        registry.register("immediate", instance="value")
        assert callbacks == [("immediate", "value")]

        # Register with factory - callback fires on first get
        registry.register("lazy", factory=lambda: "lazy_value")
        assert len(callbacks) == 1  # Not called yet

        registry.get("lazy")
        assert callbacks == [("immediate", "value"), ("lazy", "lazy_value")]

    def test_thread_safety(self):
        """Registry should be thread-safe."""
        registry = ServiceRegistry.get_instance()
        results = []
        errors = []

        def register_and_get(name: str):
            try:
                registry.register(name, instance=f"value_{name}")
                result = registry.get(name)
                results.append((name, result))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_and_get, args=(f"service_{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        for name, value in results:
            assert value == f"value_{name}"


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        ServiceRegistry.reset()

    def teardown_method(self):
        ServiceRegistry.reset()

    def test_get_registry(self):
        """get_registry should return the singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
        assert isinstance(registry1, ServiceRegistry)

    def test_register_service(self):
        """register_service should work at module level."""
        register_service("test", instance="value")
        assert get_service("test") == "value"

    def test_get_service(self):
        """get_service should work at module level."""
        register_service("test", instance="value")
        assert get_service("test") == "value"
        assert get_service("nonexistent") is None


class TestServiceEntry:
    """Tests for ServiceEntry dataclass."""

    def test_entry_creation(self):
        """ServiceEntry should store name and instance."""
        entry = ServiceEntry(name="test", instance="value")
        assert entry.name == "test"
        assert entry.instance == "value"
        assert entry.factory is None
        assert entry.initialized is False

    def test_entry_with_factory(self):
        """ServiceEntry should support factory."""
        entry = ServiceEntry(
            name="test",
            factory=lambda: "created",
            initialized=False,
        )
        assert entry.name == "test"
        assert entry.instance is None
        assert entry.factory is not None


class TestServiceName:
    """Tests for ServiceName constants."""

    def test_service_names(self):
        """ServiceName should have expected constants."""
        assert ServiceName.WORK_QUEUE == "work_queue"
        assert ServiceName.EVENT_ROUTER == "event_router"
        assert ServiceName.DAEMON_MANAGER == "daemon_manager"
        assert ServiceName.HEALTH_MANAGER == "health_manager"
        assert ServiceName.TRAINING_COORDINATOR == "training_coordinator"


class TestTypedAccessors:
    """Tests for typed accessor functions."""

    def setup_method(self):
        ServiceRegistry.reset()

    def teardown_method(self):
        ServiceRegistry.reset()

    def test_get_work_queue_returns_none_when_unregistered(self):
        """get_work_queue should return None when not registered."""
        from app.core.service_registry import get_work_queue
        assert get_work_queue() is None

    def test_get_event_router_returns_none_when_unregistered(self):
        """get_event_router should return None when not registered."""
        from app.core.service_registry import get_event_router
        assert get_event_router() is None

    def test_get_daemon_manager_returns_none_when_unregistered(self):
        """get_daemon_manager should return None when not registered."""
        from app.core.service_registry import get_daemon_manager
        assert get_daemon_manager() is None

    def test_get_health_manager_returns_none_when_unregistered(self):
        """get_health_manager should return None when not registered."""
        from app.core.service_registry import get_health_manager
        assert get_health_manager() is None
