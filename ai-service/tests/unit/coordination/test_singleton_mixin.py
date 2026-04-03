"""Tests for SingletonMixin (December 2025)."""

import threading
import time

import pytest

from app.coordination.singleton_mixin import (
    LazySingletonMixin,
    SingletonMixin,
    create_singleton_accessors,
)


class MockDaemon:
    """Mock base daemon for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


class TestSingletonMixin:
    """Test cases for SingletonMixin."""

    def setup_method(self):
        """Reset all singleton instances before each test."""
        # Clear all instances to ensure test isolation
        SingletonMixin._instances.clear()
        SingletonMixin._locks.clear()

    def test_get_instance_creates_singleton(self):
        """Should create and return a singleton instance."""

        class TestDaemon(MockDaemon, SingletonMixin):
            pass

        instance1 = TestDaemon.get_instance()
        instance2 = TestDaemon.get_instance()

        assert instance1 is instance2
        assert isinstance(instance1, TestDaemon)

    def test_get_instance_with_args(self):
        """Should pass args to first instantiation."""

        class TestDaemon(MockDaemon, SingletonMixin):
            def __init__(self, name: str = "default", value: int = 0):
                super().__init__(name)
                self.value = value

        instance = TestDaemon.get_instance("custom", value=42)

        assert instance.name == "custom"
        assert instance.value == 42

    def test_get_instance_ignores_subsequent_args(self):
        """Should ignore args on subsequent calls."""

        class TestDaemon(MockDaemon, SingletonMixin):
            def __init__(self, value: int = 0):
                super().__init__("test")
                self.value = value

        instance1 = TestDaemon.get_instance(value=42)
        instance2 = TestDaemon.get_instance(value=99)  # Should be ignored

        assert instance1 is instance2
        assert instance1.value == 42  # Original value preserved

    def test_has_instance(self):
        """Should correctly report instance existence."""

        class TestDaemon(MockDaemon, SingletonMixin):
            pass

        assert TestDaemon.has_instance() is False

        TestDaemon.get_instance()
        assert TestDaemon.has_instance() is True

        TestDaemon.reset_instance()
        assert TestDaemon.has_instance() is False

    def test_reset_instance(self):
        """Should reset the singleton instance."""

        class TestDaemon(MockDaemon, SingletonMixin):
            pass

        instance1 = TestDaemon.get_instance()
        TestDaemon.reset_instance()
        instance2 = TestDaemon.get_instance()

        assert instance1 is not instance2

    def test_reset_instance_when_no_instance(self):
        """Should handle reset when no instance exists."""

        class TestDaemon(MockDaemon, SingletonMixin):
            pass

        # Should not raise
        TestDaemon.reset_instance()

    def test_reset_instance_safe_when_stopped(self):
        """Should reset when daemon is not running."""

        class TestDaemon(MockDaemon, SingletonMixin):
            pass

        instance = TestDaemon.get_instance()
        instance._running = False

        # Should not raise
        TestDaemon.reset_instance_safe()
        assert TestDaemon.has_instance() is False

    def test_reset_instance_safe_when_running(self):
        """Should raise when daemon is still running."""

        class TestDaemon(MockDaemon, SingletonMixin):
            pass

        instance = TestDaemon.get_instance()
        instance._running = True

        with pytest.raises(RuntimeError, match="still running"):
            TestDaemon.reset_instance_safe()

    def test_separate_instances_per_class(self):
        """Should maintain separate singletons per subclass."""

        class DaemonA(MockDaemon, SingletonMixin):
            pass

        class DaemonB(MockDaemon, SingletonMixin):
            pass

        instance_a = DaemonA.get_instance("daemon_a")
        instance_b = DaemonB.get_instance("daemon_b")

        assert instance_a is not instance_b
        assert instance_a.name == "daemon_a"
        assert instance_b.name == "daemon_b"

        # Verify each class has its own singleton
        assert DaemonA.get_instance() is instance_a
        assert DaemonB.get_instance() is instance_b

    def test_thread_safety(self):
        """Should be thread-safe during concurrent access."""

        class TestDaemon(MockDaemon, SingletonMixin):
            creation_count = 0

            def __init__(self):
                super().__init__("test")
                TestDaemon.creation_count += 1
                time.sleep(0.01)  # Simulate slow initialization

        instances = []
        errors = []

        def get_instance():
            try:
                instances.append(TestDaemon.get_instance())
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to get the instance
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        assert all(i is instances[0] for i in instances)
        assert TestDaemon.creation_count == 1  # Only created once

    def test_inheritance_chain(self):
        """Should work with inheritance chains."""

        class BaseSingletonDaemon(MockDaemon, SingletonMixin):
            pass

        class ChildDaemon(BaseSingletonDaemon):
            pass

        class GrandchildDaemon(ChildDaemon):
            pass

        # Each level should have its own singleton
        base = BaseSingletonDaemon.get_instance("base")
        child = ChildDaemon.get_instance("child")
        grandchild = GrandchildDaemon.get_instance("grandchild")

        assert base is not child
        assert child is not grandchild
        assert base.name == "base"
        assert child.name == "child"
        assert grandchild.name == "grandchild"


class TestLazySingletonMixin:
    """Test cases for LazySingletonMixin."""

    def setup_method(self):
        """Reset all singleton instances before each test."""
        SingletonMixin._instances.clear()
        SingletonMixin._locks.clear()
        LazySingletonMixin._initialization_args.clear()

    def test_configure_singleton(self):
        """Should allow pre-configuration of initialization args."""

        class TestDaemon(MockDaemon, LazySingletonMixin):
            def __init__(self, config: str = "default"):
                super().__init__("test")
                self.config = config

        TestDaemon.configure_singleton(config="preconfigured")
        instance = TestDaemon.get_instance()

        assert instance.config == "preconfigured"

    def test_configured_args_override_get_instance_args(self):
        """Pre-configured args should override get_instance args."""

        class TestDaemon(MockDaemon, LazySingletonMixin):
            def __init__(self, value: int = 0):
                super().__init__("test")
                self.value = value

        TestDaemon.configure_singleton(value=100)
        instance = TestDaemon.get_instance(value=50)  # Should be ignored

        assert instance.value == 100

    def test_reset_clears_configuration(self):
        """Reset should clear pre-configured args."""

        class TestDaemon(MockDaemon, LazySingletonMixin):
            def __init__(self, value: int = 0):
                super().__init__("test")
                self.value = value

        TestDaemon.configure_singleton(value=100)
        TestDaemon.reset_instance()

        # Should use default now
        instance = TestDaemon.get_instance()
        assert instance.value == 0


class TestCreateSingletonAccessors:
    """Test cases for create_singleton_accessors helper."""

    def setup_method(self):
        """Reset all singleton instances before each test."""
        SingletonMixin._instances.clear()
        SingletonMixin._locks.clear()

    def test_creates_working_accessors(self):
        """Should create working get/reset functions."""

        class MyDaemon(MockDaemon, SingletonMixin):
            pass

        get_my_daemon, reset_my_daemon = create_singleton_accessors(MyDaemon)

        instance1 = get_my_daemon()
        instance2 = get_my_daemon()
        assert instance1 is instance2

        reset_my_daemon()
        instance3 = get_my_daemon()
        assert instance1 is not instance3

    def test_accessor_names(self):
        """Should set correct function names."""

        class MyDaemon(MockDaemon, SingletonMixin):
            pass

        get_fn, reset_fn = create_singleton_accessors(MyDaemon)

        assert get_fn.__name__ == "get_my_daemon"
        assert reset_fn.__name__ == "reset_my_daemon"

    def test_custom_accessor_names(self):
        """Should use custom names when provided."""

        class MyDaemon(MockDaemon, SingletonMixin):
            pass

        get_fn, reset_fn = create_singleton_accessors(
            MyDaemon,
            get_name="fetch_daemon",
            reset_name="clear_daemon",
        )

        assert get_fn.__name__ == "fetch_daemon"
        assert reset_fn.__name__ == "clear_daemon"

    def test_accessors_with_args(self):
        """Should pass args through to get_instance."""

        class MyDaemon(MockDaemon, SingletonMixin):
            def __init__(self, value: int = 0):
                super().__init__("test")
                self.value = value

        get_fn, reset_fn = create_singleton_accessors(MyDaemon)

        instance = get_fn(value=42)
        assert instance.value == 42

    def test_accessor_caching(self):
        """Accessors should cache the instance."""

        class MyDaemon(MockDaemon, SingletonMixin):
            pass

        get_fn, reset_fn = create_singleton_accessors(MyDaemon)

        instance1 = get_fn()

        # Reset the class singleton but not the accessor cache
        MyDaemon.reset_instance()

        # Accessor should still return cached instance
        instance2 = get_fn()
        assert instance1 is instance2

        # Full reset clears accessor cache
        reset_fn()
        instance3 = get_fn()
        assert instance1 is not instance3


class TestSingletonMixinIntegration:
    """Integration tests for SingletonMixin with real-world patterns."""

    def setup_method(self):
        """Reset all singleton instances before each test."""
        SingletonMixin._instances.clear()
        SingletonMixin._locks.clear()

    @pytest.mark.asyncio
    async def test_daemon_lifecycle(self):
        """Test singleton with async daemon lifecycle."""

        class TestDaemon(MockDaemon, SingletonMixin):
            def __init__(self):
                super().__init__("lifecycle_test")
                self.started_count = 0

            async def start(self):
                await super().start()
                self.started_count += 1

        daemon = TestDaemon.get_instance()
        await daemon.start()

        # Get same instance, verify state preserved
        same_daemon = TestDaemon.get_instance()
        assert same_daemon is daemon
        assert same_daemon.started_count == 1
        assert same_daemon.is_running

        # Clean shutdown
        await daemon.stop()
        TestDaemon.reset_instance()

    def test_with_event_subscription_mixin(self):
        """Test compatibility with the consolidated EventSubscriptionMixin."""
        from app.coordination.mixins import EventSubscriptionMixin

        class TestDaemon(MockDaemon, SingletonMixin, EventSubscriptionMixin):
            def __init__(self):
                super().__init__("combined_test")
                EventSubscriptionMixin.__init__(self)

        daemon = TestDaemon.get_instance()
        assert hasattr(daemon, "_subscription_ids")
        assert daemon._subscription_ids == []

        # Verify singleton behavior
        same_daemon = TestDaemon.get_instance()
        assert daemon is same_daemon
