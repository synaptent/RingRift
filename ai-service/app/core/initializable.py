"""Initializable Protocol for RingRift AI Service.

Provides a standard interface for components that require initialization:
- Dependency declaration and injection
- Initialization ordering via topological sort
- Status tracking (uninitialized, initializing, ready, failed)
- Async initialization support

Usage:
    from app.core.initializable import Initializable, InitializationStatus

    class DatabasePool(Initializable):
        @property
        def dependencies(self) -> List[str]:
            return []  # No dependencies

        async def initialize(self) -> None:
            self._pool = await create_pool(...)
            self._mark_ready()

    class CacheManager(Initializable):
        @property
        def dependencies(self) -> List[str]:
            return ["DatabasePool"]  # Depends on DB

        async def initialize(self) -> None:
            self._db = self._get_dependency("DatabasePool")
            self._mark_ready()

    # Automatic ordering
    registry = InitializationRegistry()
    registry.register(CacheManager())
    registry.register(DatabasePool())
    await registry.initialize_all()  # DB first, then Cache
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)

__all__ = [
    "InitializationStatus",
    "Initializable",
    "InitializationRegistry",
    "InitializationError",
    "DependencyError",
    "CircularDependencyError",
]


class InitializationStatus(Enum):
    """Status of an initializable component."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class InitializationError(Exception):
    """Base exception for initialization errors."""
    pass


class DependencyError(InitializationError):
    """Raised when a dependency cannot be resolved."""

    def __init__(self, component: str, dependency: str):
        self.component = component
        self.dependency = dependency
        super().__init__(f"Component '{component}' depends on '{dependency}' which is not registered")


class CircularDependencyError(InitializationError):
    """Raised when circular dependencies are detected."""

    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}")


@dataclass
class InitializationResult:
    """Result of an initialization attempt."""
    component_name: str
    status: InitializationStatus
    duration_ms: float
    error: Optional[Exception] = None


class Initializable(ABC):
    """Abstract base class for components that require initialization.

    Subclasses should:
    1. Define dependencies via the `dependencies` property
    2. Implement `initialize()` to set up the component
    3. Optionally implement `shutdown()` for cleanup
    4. Call `_mark_ready()` when initialization completes

    Example:
        class MyService(Initializable):
            @property
            def name(self) -> str:
                return "MyService"

            @property
            def dependencies(self) -> List[str]:
                return ["DatabasePool", "CacheManager"]

            async def initialize(self) -> None:
                self._db = self._get_dependency("DatabasePool")
                self._cache = self._get_dependency("CacheManager")
                await self._setup()
                self._mark_ready()
    """

    def __init__(self):
        self._status = InitializationStatus.UNINITIALIZED
        self._error: Optional[Exception] = None
        self._dependencies_resolved: Dict[str, Any] = {}
        self._on_status_change: Optional[Callable[[InitializationStatus], None]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this component."""
        pass

    @property
    def dependencies(self) -> List[str]:
        """List of component names this depends on.

        Override to declare dependencies. Default is no dependencies.
        """
        return []

    @property
    def status(self) -> InitializationStatus:
        """Current initialization status."""
        return self._status

    @property
    def is_ready(self) -> bool:
        """Check if component is ready for use."""
        return self._status == InitializationStatus.READY

    @property
    def error(self) -> Optional[Exception]:
        """Error that caused initialization failure, if any."""
        return self._error

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component.

        Dependencies are available via `_get_dependency()`.
        Call `_mark_ready()` when initialization is complete.

        Raises:
            InitializationError: If initialization fails
        """
        pass

    async def shutdown(self) -> None:
        """Shutdown the component and release resources.

        Override to implement cleanup. Default does nothing.
        """
        pass

    def _set_dependency(self, name: str, instance: Any) -> None:
        """Set a resolved dependency (called by registry)."""
        self._dependencies_resolved[name] = instance

    def _get_dependency(self, name: str) -> Any:
        """Get a resolved dependency.

        Args:
            name: Dependency component name

        Returns:
            The dependency instance

        Raises:
            DependencyError: If dependency not resolved
        """
        if name not in self._dependencies_resolved:
            raise DependencyError(self.name, name)
        return self._dependencies_resolved[name]

    def _mark_ready(self) -> None:
        """Mark component as ready (call from initialize())."""
        self._set_status(InitializationStatus.READY)

    def _mark_failed(self, error: Exception) -> None:
        """Mark component as failed (call from initialize() on error)."""
        self._error = error
        self._set_status(InitializationStatus.FAILED)

    def _set_status(self, status: InitializationStatus) -> None:
        """Set the status and notify listener."""
        old_status = self._status
        self._status = status
        if self._on_status_change and old_status != status:
            self._on_status_change(status)
        logger.debug(f"{self.name}: {old_status.value} -> {status.value}")


class InitializationRegistry:
    """Registry for managing component initialization order.

    Handles:
    - Component registration
    - Dependency resolution via topological sort
    - Parallel initialization where possible
    - Graceful shutdown in reverse order

    Example:
        registry = InitializationRegistry()
        registry.register(DatabasePool())
        registry.register(CacheManager())  # depends on DB
        registry.register(ApiServer())     # depends on Cache

        await registry.initialize_all()
        # Initializes: DB -> Cache -> Api (respecting deps)

        await registry.shutdown_all()
        # Shuts down: Api -> Cache -> DB (reverse order)
    """

    def __init__(self):
        self._components: Dict[str, Initializable] = {}
        self._initialization_order: List[str] = []
        self._results: List[InitializationResult] = []

    def register(self, component: Initializable) -> None:
        """Register a component for initialization.

        Args:
            component: The component to register

        Raises:
            ValueError: If component with same name already registered
        """
        if component.name in self._components:
            raise ValueError(f"Component '{component.name}' already registered")
        self._components[component.name] = component
        logger.debug(f"Registered component: {component.name}")

    def get(self, name: str) -> Optional[Initializable]:
        """Get a registered component by name."""
        return self._components.get(name)

    def get_required(self, name: str) -> Initializable:
        """Get a registered component, raising if not found."""
        component = self._components.get(name)
        if component is None:
            raise KeyError(f"Component '{name}' not registered")
        return component

    @property
    def components(self) -> Dict[str, Initializable]:
        """All registered components."""
        return dict(self._components)

    @property
    def results(self) -> List[InitializationResult]:
        """Results from last initialization."""
        return list(self._results)

    def compute_order(self) -> List[str]:
        """Compute initialization order via topological sort.

        Returns:
            List of component names in initialization order

        Raises:
            DependencyError: If a dependency is not registered
            CircularDependencyError: If circular dependencies exist
        """
        # Validate all dependencies exist
        for name, component in self._components.items():
            for dep in component.dependencies:
                if dep not in self._components:
                    raise DependencyError(name, dep)

        # Topological sort using Kahn's algorithm
        in_degree: Dict[str, int] = {name: 0 for name in self._components}
        graph: Dict[str, List[str]] = {name: [] for name in self._components}

        for name, component in self._components.items():
            for dep in component.dependencies:
                graph[dep].append(name)
                in_degree[name] += 1

        # Start with components that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order: List[str] = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            name = queue.pop(0)
            order.append(name)

            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(order) != len(self._components):
            # Find cycle for error message
            remaining = set(self._components.keys()) - set(order)
            cycle = self._find_cycle(remaining)
            raise CircularDependencyError(cycle)

        self._initialization_order = order
        return order

    def _find_cycle(self, nodes: Set[str]) -> List[str]:
        """Find a cycle in the dependency graph."""
        visited: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            component = self._components[node]
            for dep in component.dependencies:
                if dep in nodes:
                    result = dfs(dep)
                    if result:
                        return result

            path.pop()
            return None

        for node in nodes:
            result = dfs(node)
            if result:
                return result

        return list(nodes)  # Fallback

    async def initialize_all(
        self,
        parallel: bool = False,
        stop_on_failure: bool = True,
    ) -> List[InitializationResult]:
        """Initialize all registered components.

        Args:
            parallel: Initialize independent components in parallel
            stop_on_failure: Stop initialization if any component fails

        Returns:
            List of initialization results

        Raises:
            InitializationError: If initialization fails and stop_on_failure=True
        """
        order = self.compute_order()
        self._results = []

        logger.info(f"Initializing {len(order)} components: {' -> '.join(order)}")

        # Resolve dependencies before initialization
        for name in order:
            component = self._components[name]
            for dep_name in component.dependencies:
                dep = self._components[dep_name]
                component._set_dependency(dep_name, dep)

        if parallel:
            await self._initialize_parallel(order, stop_on_failure)
        else:
            await self._initialize_sequential(order, stop_on_failure)

        # Summary
        succeeded = sum(1 for r in self._results if r.status == InitializationStatus.READY)
        failed = sum(1 for r in self._results if r.status == InitializationStatus.FAILED)
        logger.info(f"Initialization complete: {succeeded} succeeded, {failed} failed")

        return self._results

    async def _initialize_sequential(
        self,
        order: List[str],
        stop_on_failure: bool,
    ) -> None:
        """Initialize components sequentially."""
        import time

        for name in order:
            component = self._components[name]
            start = time.time()

            try:
                component._set_status(InitializationStatus.INITIALIZING)
                await component.initialize()

                # Ensure status was set
                if component.status == InitializationStatus.INITIALIZING:
                    component._mark_ready()

                duration = (time.time() - start) * 1000
                self._results.append(InitializationResult(
                    component_name=name,
                    status=component.status,
                    duration_ms=duration,
                ))
                logger.info(f"Initialized {name} in {duration:.1f}ms")

            except Exception as e:
                duration = (time.time() - start) * 1000
                component._mark_failed(e)
                self._results.append(InitializationResult(
                    component_name=name,
                    status=InitializationStatus.FAILED,
                    duration_ms=duration,
                    error=e,
                ))
                logger.error(f"Failed to initialize {name}: {e}")

                if stop_on_failure:
                    raise InitializationError(f"Failed to initialize {name}: {e}") from e

    async def _initialize_parallel(
        self,
        order: List[str],
        stop_on_failure: bool,
    ) -> None:
        """Initialize independent components in parallel."""
        import time

        initialized: Set[str] = set()
        pending = set(order)

        while pending:
            # Find components whose dependencies are all initialized
            ready = []
            for name in pending:
                component = self._components[name]
                if all(dep in initialized for dep in component.dependencies):
                    ready.append(name)

            if not ready:
                # Should not happen if order is valid
                break

            # Initialize ready components in parallel
            async def init_one(name: str) -> InitializationResult:
                component = self._components[name]
                start = time.time()

                try:
                    component._set_status(InitializationStatus.INITIALIZING)
                    await component.initialize()

                    if component.status == InitializationStatus.INITIALIZING:
                        component._mark_ready()

                    duration = (time.time() - start) * 1000
                    logger.info(f"Initialized {name} in {duration:.1f}ms")
                    return InitializationResult(
                        component_name=name,
                        status=component.status,
                        duration_ms=duration,
                    )
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    component._mark_failed(e)
                    logger.error(f"Failed to initialize {name}: {e}")
                    return InitializationResult(
                        component_name=name,
                        status=InitializationStatus.FAILED,
                        duration_ms=duration,
                        error=e,
                    )

            results = await asyncio.gather(
                *[init_one(name) for name in ready],
                return_exceptions=False,
            )

            for result in results:
                self._results.append(result)
                if result.status == InitializationStatus.READY:
                    initialized.add(result.component_name)
                    pending.discard(result.component_name)
                elif stop_on_failure:
                    raise InitializationError(
                        f"Failed to initialize {result.component_name}: {result.error}"
                    )
                else:
                    pending.discard(result.component_name)

    async def shutdown_all(self, reverse: bool = True) -> None:
        """Shutdown all initialized components.

        Args:
            reverse: Shutdown in reverse initialization order (default True)
        """
        order = list(self._initialization_order)
        if reverse:
            order.reverse()

        logger.info(f"Shutting down {len(order)} components")

        for name in order:
            component = self._components.get(name)
            if component and component.status == InitializationStatus.READY:
                try:
                    component._set_status(InitializationStatus.SHUTTING_DOWN)
                    await component.shutdown()
                    component._set_status(InitializationStatus.SHUTDOWN)
                    logger.debug(f"Shut down {name}")
                except Exception as e:
                    logger.error(f"Error shutting down {name}: {e}")

        logger.info("Shutdown complete")


# Convenience decorator for simple components
def initializable(
    name: str,
    dependencies: Optional[List[str]] = None,
) -> Callable[[Type], Type]:
    """Decorator to make a class initializable.

    Usage:
        @initializable("MyService", dependencies=["Database"])
        class MyService:
            async def initialize(self) -> None:
                ...
    """
    def decorator(cls: Type) -> Type:
        original_init = cls.__init__ if hasattr(cls, '__init__') else None

        def new_init(self, *args, **kwargs):
            Initializable.__init__(self)
            if original_init:
                original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls.name = property(lambda self: name)
        cls.dependencies = property(lambda self: dependencies or [])

        # Ensure it's a proper subclass
        if not issubclass(cls, Initializable):
            # Mix in Initializable
            cls = type(cls.__name__, (cls, Initializable), dict(cls.__dict__))

        return cls

    return decorator
