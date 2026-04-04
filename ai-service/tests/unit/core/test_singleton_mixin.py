"""Tests for singleton patterns in app.core.singleton_mixin."""

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from app.core.singleton_mixin import (
        SingletonMeta,
        SingletonMixin,
        ThreadSafeSingletonMixin,
        singleton,
    )


class TestSingletonMeta:
    """Tests for SingletonMeta metaclass."""

    def test_creates_single_instance(self) -> None:
        """Metaclass creates exactly one instance."""

        class Service(metaclass=SingletonMeta):
            def __init__(self) -> None:
                self.created_at = time.time()

        s1 = Service()
        s2 = Service()
        assert s1 is s2

    def test_preserves_first_init_args(self) -> None:
        """First initialization arguments are preserved."""

        class ConfigService(metaclass=SingletonMeta):
            def __init__(self, name: str = "default") -> None:
                self.name = name

        ConfigService.reset_instance()  # Ensure clean state
        s1 = ConfigService("first")
        s2 = ConfigService("second")
        assert s1.name == "first"
        assert s2.name == "first"
        ConfigService.reset_instance()  # Cleanup

    def test_get_instance_method(self) -> None:
        """get_instance() returns the singleton."""

        class Registry(metaclass=SingletonMeta):
            pass

        Registry.reset_instance()
        r1 = Registry.get_instance()
        r2 = Registry.get_instance()
        assert r1 is r2
        Registry.reset_instance()

    def test_reset_instance(self) -> None:
        """reset_instance() clears the singleton."""

        class Cache(metaclass=SingletonMeta):
            def __init__(self) -> None:
                self.data: dict = {}

        Cache.reset_instance()
        c1 = Cache()
        c1.data["key"] = "value"
        Cache.reset_instance()
        c2 = Cache()
        assert c2.data == {}
        assert c1 is not c2
        Cache.reset_instance()

    def test_has_instance(self) -> None:
        """has_instance() reports singleton existence."""

        class Monitor(metaclass=SingletonMeta):
            pass

        Monitor.reset_instance()
        assert not Monitor.has_instance()
        Monitor()
        assert Monitor.has_instance()
        Monitor.reset_instance()
        assert not Monitor.has_instance()

    def test_thread_safety(self) -> None:
        """Singleton is thread-safe under concurrent access."""

        class Counter(metaclass=SingletonMeta):
            def __init__(self) -> None:
                self.count = 0
                self._lock = threading.Lock()

            def increment(self) -> None:
                with self._lock:
                    self.count += 1

        Counter.reset_instance()
        instances: list = []

        def get_and_increment() -> None:
            instance = Counter()
            instances.append(instance)
            instance.increment()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_and_increment) for _ in range(100)]
            for f in futures:
                f.result()

        # All instances should be the same object
        assert len(set(id(i) for i in instances)) == 1
        assert Counter().count == 100
        Counter.reset_instance()

    def test_cleanup_called_on_reset(self) -> None:
        """_cleanup() is called when reset_instance() is invoked."""
        cleanup_called = []

        class Cleanable(metaclass=SingletonMeta):
            def _cleanup(self) -> None:
                cleanup_called.append(True)

        Cleanable.reset_instance()
        Cleanable()
        Cleanable.reset_instance()
        assert len(cleanup_called) == 1


class TestSingletonMixin:
    """Tests for SingletonMixin class."""

    def test_get_or_create_instance(self) -> None:
        """_get_or_create_instance() returns singleton."""

        class MyService(SingletonMixin):
            _instance: "MyService | None" = None
            _lock = threading.RLock()

        s1 = MyService._get_or_create_instance()
        s2 = MyService._get_or_create_instance()
        assert s1 is s2
        MyService.reset_instance()

    def test_reset_saves_state(self) -> None:
        """reset_instance() calls _save_state() if available."""
        save_called = []

        class Stateful(SingletonMixin):
            _instance: "Stateful | None" = None
            _lock = threading.RLock()

            def _save_state(self) -> None:
                save_called.append(True)

        Stateful.get_instance()
        Stateful.reset_instance()
        assert len(save_called) == 1

    def test_subclass_isolation(self) -> None:
        """Each subclass has its own singleton instance."""

        class BaseService(SingletonMixin):
            pass

        class ServiceA(BaseService):
            pass

        class ServiceB(BaseService):
            pass

        a = ServiceA.get_instance()
        b = ServiceB.get_instance()
        assert a is not b
        assert type(a) is ServiceA
        assert type(b) is ServiceB
        ServiceA.reset_instance()
        ServiceB.reset_instance()


class TestThreadSafeSingletonMixin:
    """Tests for ThreadSafeSingletonMixin."""

    def test_double_checked_locking(self) -> None:
        """Fast path avoids lock acquisition."""

        class FastRegistry(ThreadSafeSingletonMixin):
            _instance: "FastRegistry | None" = None
            _lock = threading.RLock()

        # First call takes slow path
        r1 = FastRegistry.get_instance()
        # Subsequent calls should use fast path
        r2 = FastRegistry.get_instance()
        r3 = FastRegistry.get_instance()
        assert r1 is r2 is r3
        FastRegistry.reset_instance()


class TestSingletonDecorator:
    """Tests for @singleton decorator."""

    def test_basic_singleton(self) -> None:
        """@singleton creates single instance."""

        @singleton
        class AppConfig:
            def __init__(self) -> None:
                self.debug = False

        AppConfig.reset_instance()
        c1 = AppConfig()
        c2 = AppConfig()
        assert c1 is c2
        AppConfig.reset_instance()

    def test_init_called_once(self) -> None:
        """__init__ is called exactly once."""
        init_count = [0]

        @singleton
        class Counter:
            def __init__(self) -> None:
                init_count[0] += 1

        Counter.reset_instance()
        Counter()
        Counter()
        Counter()
        assert init_count[0] == 1
        Counter.reset_instance()

    def test_has_helper_methods(self) -> None:
        """Decorator adds helper methods."""

        @singleton
        class Helper:
            pass

        Helper.reset_instance()
        assert hasattr(Helper, "get_instance")
        assert hasattr(Helper, "has_instance")
        assert hasattr(Helper, "reset_instance")
        assert not Helper.has_instance()
        Helper()
        assert Helper.has_instance()
        Helper.reset_instance()


class TestMetricCatalogIntegration:
    """Integration test with actual MetricCatalog."""

    def test_catalog_is_singleton(self) -> None:
        """MetricCatalog uses SingletonMixin correctly."""
        from app.metrics.catalog import MetricCatalog

        MetricCatalog.reset_instance()
        c1 = MetricCatalog.get_instance()
        c2 = MetricCatalog.get_instance()
        assert c1 is c2
        MetricCatalog.reset_instance()

    def test_catalog_metrics_registered(self) -> None:
        """MetricCatalog registers metrics on first access."""
        from app.metrics.catalog import MetricCatalog

        MetricCatalog.reset_instance()
        catalog = MetricCatalog.get_instance()
        # Should have registered metrics
        all_metrics = catalog.list_all()
        assert len(all_metrics) > 0
        MetricCatalog.reset_instance()
