"""Coordinator Dependency Management (December 2025).

Provides:
- Dependency graph for coordinators
- Circular dependency detection
- Topological ordering for initialization
- Dependency validation

Usage:
    from app.coordination.coordinator_dependencies import (
        CoordinatorDependencyGraph,
        validate_dependencies,
        get_initialization_order,
    )

    # Validate no circular dependencies
    if not validate_dependencies():
        raise RuntimeError("Circular dependency detected!")

    # Get safe initialization order
    order = get_initialization_order()
    for coordinator_name in order:
        initialize(coordinator_name)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorDependency:
    """Represents a coordinator and its dependencies."""

    name: str
    emits: set[str] = field(default_factory=set)  # Event types emitted
    subscribes: set[str] = field(default_factory=set)  # Event types subscribed to
    depends_on: set[str] = field(default_factory=set)  # Direct coordinator dependencies


# Known coordinator dependencies (December 2025)
# Format: coordinator -> (emits, subscribes, depends_on)
COORDINATOR_REGISTRY: dict[str, CoordinatorDependency] = {
    "task_lifecycle": CoordinatorDependency(
        name="task_lifecycle",
        emits={"task_orphaned", "task_completed", "task_failed"},
        subscribes={"task_spawned", "task_heartbeat", "host_online", "host_offline", "node_recovered"},
        depends_on=set(),  # Foundational - no dependencies
    ),
    "resources": CoordinatorDependency(
        name="resources",
        emits={"backpressure_activated", "backpressure_released", "resource_constraint_detected"},
        subscribes={"node_capacity_updated"},
        depends_on=set(),  # Foundational - no dependencies
    ),
    "cache": CoordinatorDependency(
        name="cache",
        emits={"cache_invalidated"},
        subscribes=set(),
        depends_on=set(),  # Foundational - no dependencies
    ),
    "selfplay": CoordinatorDependency(
        name="selfplay",
        emits={"selfplay_complete", "task_spawned"},
        subscribes={
            "task_spawned", "task_completed", "task_failed",
            "backpressure_activated", "backpressure_released",
            "resource_constraint_detected", "regression_detected",
        },
        depends_on={"task_lifecycle", "resources"},
    ),
    "pipeline": CoordinatorDependency(
        name="pipeline",
        emits={"training_started", "training_complete", "iteration_complete"},
        subscribes={
            "selfplay_complete", "training_complete", "sync_complete",
            "quality_distribution_changed", "cache_invalidated",
            "cmaes_triggered", "nas_triggered",
        },
        depends_on={"selfplay", "cache"},
    ),
    "optimization": CoordinatorDependency(
        name="optimization",
        emits={"cmaes_triggered", "nas_triggered", "hyperparameter_updated"},
        subscribes={"plateau_detected", "training_complete"},
        depends_on={"metrics"},
    ),
    "metrics": CoordinatorDependency(
        name="metrics",
        emits={"plateau_detected", "regression_detected", "elo_significant_change"},
        subscribes={"training_complete", "evaluation_complete"},
        depends_on={"pipeline"},
    ),
}


class CoordinatorDependencyGraph:
    """Graph for analyzing coordinator dependencies."""

    def __init__(self):
        self._dependencies: dict[str, CoordinatorDependency] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)  # coordinator -> depends on
        self._reverse: dict[str, set[str]] = defaultdict(set)  # coordinator -> depended by

    def add_coordinator(self, dep: CoordinatorDependency) -> None:
        """Add a coordinator to the graph."""
        self._dependencies[dep.name] = dep

        for depends_on in dep.depends_on:
            self._adjacency[dep.name].add(depends_on)
            self._reverse[depends_on].add(dep.name)

    def build_from_registry(self) -> None:
        """Build graph from the global registry."""
        for dep in COORDINATOR_REGISTRY.values():
            self.add_coordinator(dep)

    def detect_cycles(self) -> list[list[str]]:
        """Detect circular dependencies using DFS.

        Returns:
            List of cycles found (each cycle is a list of coordinator names)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle_start = path.index(neighbor)
                    cycle = [*path[cycle_start:], neighbor]
                    cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in self._dependencies:
            if node not in visited:
                dfs(node)

        return cycles

    def topological_sort(self) -> tuple[list[str], bool]:
        """Get initialization order using Kahn's algorithm.

        Returns:
            (ordered_list, success) - success is False if cycle exists
        """
        in_degree = defaultdict(int)
        for node in self._dependencies:
            in_degree[node] = 0

        for node in self._dependencies:
            for _dep in self._adjacency.get(node, set()):
                in_degree[node] += 1

        # Start with nodes that have no dependencies
        queue = [n for n in self._dependencies if in_degree[n] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self._reverse.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._dependencies):
            return (result, False)  # Cycle exists

        return (result, True)

    def get_dependents(self, coordinator: str) -> set[str]:
        """Get coordinators that depend on the given coordinator."""
        return self._reverse.get(coordinator, set())

    def get_dependencies(self, coordinator: str) -> set[str]:
        """Get coordinators that the given coordinator depends on."""
        return self._adjacency.get(coordinator, set())

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the dependency graph.

        Returns:
            (valid, issues) - issues is list of problem descriptions
        """
        issues = []

        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            for cycle in cycles:
                issues.append(f"Circular dependency: {' -> '.join(cycle)}")

        # Check for missing dependencies
        all_coordinators = set(self._dependencies.keys())
        for name, dep in self._dependencies.items():
            missing = dep.depends_on - all_coordinators
            if missing:
                issues.append(f"{name} depends on unknown coordinators: {missing}")

        return (len(issues) == 0, issues)


# Global graph instance
_dependency_graph: CoordinatorDependencyGraph | None = None


def get_dependency_graph() -> CoordinatorDependencyGraph:
    """Get the global dependency graph (lazy initialization)."""
    global _dependency_graph
    if _dependency_graph is None:
        _dependency_graph = CoordinatorDependencyGraph()
        _dependency_graph.build_from_registry()
    return _dependency_graph


def reset_dependency_graph() -> None:
    """Reset the global dependency graph (for testing)."""
    global _dependency_graph
    _dependency_graph = None


def validate_dependencies() -> tuple[bool, list[str]]:
    """Validate coordinator dependencies.

    Returns:
        (valid, issues) - issues is list of problem descriptions
    """
    graph = get_dependency_graph()
    return graph.validate()


def get_initialization_order() -> list[str]:
    """Get safe initialization order for coordinators.

    Returns:
        List of coordinator names in dependency order

    Raises:
        RuntimeError: If circular dependency exists
    """
    graph = get_dependency_graph()
    order, success = graph.topological_sort()

    if not success:
        cycles = graph.detect_cycles()
        raise RuntimeError(f"Cannot determine initialization order - circular dependencies: {cycles}")

    return order


def check_event_chain_cycles() -> list[list[str]]:
    """Check for cycles in event emission/subscription chains.

    This is more comprehensive than direct dependency cycles - it checks
    if A emits X, B subscribes to X and emits Y, C subscribes to Y and emits Z,
    and A subscribes to Z (forming a cycle through events).

    Returns:
        List of event chain cycles
    """
    graph = get_dependency_graph()

    # Build event-based adjacency
    event_to_emitters: dict[str, set[str]] = defaultdict(set)
    event_to_subscribers: dict[str, set[str]] = defaultdict(set)

    for name, dep in graph._dependencies.items():
        for event in dep.emits:
            event_to_emitters[event].add(name)
        for event in dep.subscribes:
            event_to_subscribers[event].add(name)

    # Build coordinator adjacency through events
    event_adjacency: dict[str, set[str]] = defaultdict(set)
    for event, subscribers in event_to_subscribers.items():
        emitters = event_to_emitters.get(event, set())
        for subscriber in subscribers:
            for emitter in emitters:
                if subscriber != emitter:
                    event_adjacency[subscriber].add(emitter)

    # Detect cycles in event-based graph
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in event_adjacency.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycle = [*path[cycle_start:], neighbor]
                cycles.append(cycle)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    for node in graph._dependencies:
        if node not in visited:
            dfs(node)

    return cycles


__all__ = [
    "COORDINATOR_REGISTRY",
    "CoordinatorDependency",
    "CoordinatorDependencyGraph",
    "check_event_chain_cycles",
    "get_dependency_graph",
    "get_initialization_order",
    "reset_dependency_graph",
    "validate_dependencies",
]
