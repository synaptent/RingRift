"""Singleton Registry - Auto-discovery and reset for all singletons (December 2025).

Provides centralized management of all SingletonMixin-based singletons for testing.

Problem Solved:
    Tests fail due to state pollution from singleton instances persisting across tests.
    Previously, conftest.py only reset 8 hardcoded singletons, but 60+ exist in the codebase.
    This registry auto-discovers ALL singletons via SingletonMixin._instances.

Usage:
    # In conftest.py (autouse fixture)
    @pytest.fixture(autouse=True)
    def reset_all_singletons():
        from app.coordination.singleton_registry import SingletonRegistry
        SingletonRegistry.reset_all_sync()
        yield
        SingletonRegistry.reset_all_sync()

    # In tests requiring async cleanup
    @pytest.fixture
    async def clean_singletons():
        from app.coordination.singleton_registry import SingletonRegistry
        await SingletonRegistry.reset_all_async()
        yield
        await SingletonRegistry.reset_all_async()

    # Query what singletons exist
    from app.coordination.singleton_registry import SingletonRegistry
    for cls, instance in SingletonRegistry.get_all_singletons().items():
        print(f"{cls.__name__}: {instance}")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SingletonRegistry:
    """Registry for auto-discovering and resetting all SingletonMixin singletons.

    This class provides static methods to:
    - Get all current singleton instances
    - Reset all singletons synchronously (for basic cleanup)
    - Reset all singletons asynchronously (for daemons with async cleanup)

    Thread Safety:
        All operations are thread-safe, using the per-class locks from SingletonMixin.
    """

    @staticmethod
    def get_all_singletons() -> dict[type, Any]:
        """Get a copy of all current singleton instances.

        Returns:
            Dict mapping singleton class types to their instances.
            Returns a copy to prevent modification of the internal dict.

        Example:
            singletons = SingletonRegistry.get_all_singletons()
            print(f"Found {len(singletons)} singletons")
            for cls, instance in singletons.items():
                print(f"  {cls.__name__}")
        """
        from app.coordination.singleton_mixin import SingletonMixin

        return dict(SingletonMixin._instances)

    @staticmethod
    def get_singleton_count() -> int:
        """Get the count of active singletons.

        Returns:
            Number of singleton instances currently tracked.
        """
        from app.coordination.singleton_mixin import SingletonMixin

        return len(SingletonMixin._instances)

    @staticmethod
    def has_singleton(cls: type) -> bool:
        """Check if a specific class has an active singleton.

        Args:
            cls: The class to check.

        Returns:
            True if the class has an active singleton instance.
        """
        from app.coordination.singleton_mixin import SingletonMixin

        return cls in SingletonMixin._instances

    @staticmethod
    def reset_all_sync() -> int:
        """Reset all singleton instances synchronously.

        This calls reset_instance() on each singleton class, which removes
        the instance from the registry. It does NOT call any cleanup methods
        on the instances themselves - call stop() explicitly if needed.

        Returns:
            Number of singletons that were reset.

        Example:
            # In test setup/teardown
            count = SingletonRegistry.reset_all_sync()
            print(f"Reset {count} singletons")
        """
        from app.coordination.singleton_mixin import SingletonMixin

        # Get list of classes to reset (copy to avoid modification during iteration)
        classes_to_reset = list(SingletonMixin._instances.keys())
        count = 0

        for cls in classes_to_reset:
            try:
                # Check if class has custom reset_instance
                if hasattr(cls, "reset_instance"):
                    cls.reset_instance()
                    count += 1
                    logger.debug(f"Reset singleton: {cls.__name__}")
                else:
                    # Fallback: directly remove from instances dict
                    with cls._get_lock():
                        if cls in SingletonMixin._instances:
                            del SingletonMixin._instances[cls]
                            count += 1
                            logger.debug(f"Removed singleton: {cls.__name__}")
            except Exception as e:
                logger.warning(f"Failed to reset {cls.__name__}: {e}")

        return count

    @staticmethod
    async def reset_all_async() -> int:
        """Reset all singleton instances with async cleanup support.

        This method handles singletons that require async cleanup (like daemons).
        It:
        1. Calls stop() on any instances that have it (async or sync)
        2. Cancels any _health_task attributes
        3. Then calls reset_instance() to remove from registry

        Returns:
            Number of singletons that were reset.

        Example:
            # In async test fixture
            async def cleanup_singletons():
                count = await SingletonRegistry.reset_all_async()
                print(f"Async-reset {count} singletons")
        """
        from app.coordination.singleton_mixin import SingletonMixin

        # Get list of classes and instances to reset
        instances_to_reset = list(SingletonMixin._instances.items())
        count = 0

        for cls, instance in instances_to_reset:
            try:
                # Step 1: Stop the instance if it has a stop method
                if hasattr(instance, "stop"):
                    stop_method = getattr(instance, "stop")
                    if asyncio.iscoroutinefunction(stop_method):
                        try:
                            await asyncio.wait_for(stop_method(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout stopping {cls.__name__}")
                        except Exception as e:
                            logger.debug(f"Error stopping {cls.__name__}: {e}")
                    else:
                        try:
                            stop_method()
                        except Exception as e:
                            logger.debug(f"Error stopping {cls.__name__}: {e}")

                # Step 2: Cancel health task if present (DaemonManager, etc.)
                if hasattr(instance, "_health_task") and instance._health_task:
                    task = instance._health_task
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(task), timeout=1.0
                            )
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass

                # Step 3: Clear internal state if the instance has clear methods
                for clear_method_name in [
                    "_clear_internal_state",
                    "clear",
                    "_reset_state",
                ]:
                    if hasattr(instance, clear_method_name):
                        try:
                            clear_method = getattr(instance, clear_method_name)
                            if asyncio.iscoroutinefunction(clear_method):
                                await clear_method()
                            else:
                                clear_method()
                        except Exception as e:
                            logger.debug(
                                f"Error clearing {cls.__name__}.{clear_method_name}: {e}"
                            )
                        break

                # Step 4: Reset the singleton
                if hasattr(cls, "reset_instance"):
                    cls.reset_instance()
                else:
                    with cls._get_lock():
                        if cls in SingletonMixin._instances:
                            del SingletonMixin._instances[cls]

                count += 1
                logger.debug(f"Async-reset singleton: {cls.__name__}")

            except Exception as e:
                logger.warning(f"Failed to async-reset {cls.__name__}: {e}")

        return count

    @staticmethod
    def get_singletons_by_category() -> dict[str, list[type]]:
        """Group singletons by category based on module path.

        Returns:
            Dict mapping category names to lists of singleton classes.
            Categories: coordination, training, ai, distributed, core, other

        Example:
            by_category = SingletonRegistry.get_singletons_by_category()
            for category, classes in by_category.items():
                print(f"{category}: {len(classes)} singletons")
        """
        from app.coordination.singleton_mixin import SingletonMixin

        categories: dict[str, list[type]] = {
            "coordination": [],
            "training": [],
            "ai": [],
            "distributed": [],
            "core": [],
            "other": [],
        }

        for cls in SingletonMixin._instances.keys():
            module = cls.__module__
            if "coordination" in module:
                categories["coordination"].append(cls)
            elif "training" in module:
                categories["training"].append(cls)
            elif ".ai." in module or module.endswith(".ai"):
                categories["ai"].append(cls)
            elif "distributed" in module:
                categories["distributed"].append(cls)
            elif "core" in module:
                categories["core"].append(cls)
            else:
                categories["other"].append(cls)

        return categories

    @staticmethod
    def get_running_daemons() -> list[tuple[type, Any]]:
        """Get singletons that appear to be running daemons.

        Returns:
            List of (class, instance) tuples for singletons with
            _running=True or is_running=True attributes.
        """
        from app.coordination.singleton_mixin import SingletonMixin

        running = []
        for cls, instance in SingletonMixin._instances.items():
            is_running = getattr(instance, "is_running", None)
            if is_running is None:
                is_running = getattr(instance, "_running", False)
            if is_running:
                running.append((cls, instance))

        return running

    @staticmethod
    def stop_all_daemons_sync() -> int:
        """Stop all running daemons synchronously.

        Calls stop() on any singletons that have _running=True or is_running=True.
        Does NOT reset the singletons - call reset_all_sync() after if needed.

        Returns:
            Number of daemons stopped.
        """
        stopped = 0
        for cls, instance in SingletonRegistry.get_running_daemons():
            if hasattr(instance, "stop"):
                try:
                    stop_method = getattr(instance, "stop")
                    if not asyncio.iscoroutinefunction(stop_method):
                        stop_method()
                        stopped += 1
                        logger.debug(f"Stopped daemon: {cls.__name__}")
                except Exception as e:
                    logger.warning(f"Failed to stop {cls.__name__}: {e}")

        return stopped

    @staticmethod
    async def stop_all_daemons_async() -> int:
        """Stop all running daemons asynchronously.

        Calls stop() on any singletons that have _running=True or is_running=True.
        Handles both sync and async stop methods.

        Returns:
            Number of daemons stopped.
        """
        stopped = 0
        for cls, instance in SingletonRegistry.get_running_daemons():
            if hasattr(instance, "stop"):
                try:
                    stop_method = getattr(instance, "stop")
                    if asyncio.iscoroutinefunction(stop_method):
                        await asyncio.wait_for(stop_method(), timeout=5.0)
                    else:
                        stop_method()
                    stopped += 1
                    logger.debug(f"Stopped daemon: {cls.__name__}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout stopping daemon: {cls.__name__}")
                except Exception as e:
                    logger.warning(f"Failed to stop {cls.__name__}: {e}")

        return stopped


# Convenience functions for common operations
def get_singleton_count() -> int:
    """Get the count of active singletons."""
    return SingletonRegistry.get_singleton_count()


def reset_all_singletons() -> int:
    """Reset all singletons synchronously."""
    return SingletonRegistry.reset_all_sync()


async def reset_all_singletons_async() -> int:
    """Reset all singletons with async cleanup."""
    return await SingletonRegistry.reset_all_async()


__all__ = [
    "SingletonRegistry",
    "get_singleton_count",
    "reset_all_singletons",
    "reset_all_singletons_async",
]
