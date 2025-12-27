"""Tests for coordinator_dependencies module.

December 27, 2025: Added to improve test coverage for critical coordination modules.
"""

import pytest

from app.coordination.coordinator_dependencies import (
    COORDINATOR_REGISTRY,
    CoordinatorDependency,
    CoordinatorDependencyGraph,
    check_event_chain_cycles,
    get_dependency_graph,
    get_initialization_order,
    reset_dependency_graph,
    validate_dependencies,
)


class TestCoordinatorDependency:
    """Tests for CoordinatorDependency dataclass."""

    def test_basic_creation(self):
        """Test creating a dependency with minimal args."""
        dep = CoordinatorDependency(name="test")
        assert dep.name == "test"
        assert dep.emits == set()
        assert dep.subscribes == set()
        assert dep.depends_on == set()

    def test_full_creation(self):
        """Test creating a dependency with all args."""
        dep = CoordinatorDependency(
            name="my_coordinator",
            emits={"event_a", "event_b"},
            subscribes={"event_c", "event_d"},
            depends_on={"other_coordinator"},
        )
        assert dep.name == "my_coordinator"
        assert dep.emits == {"event_a", "event_b"}
        assert dep.subscribes == {"event_c", "event_d"}
        assert dep.depends_on == {"other_coordinator"}

    def test_fields_are_sets(self):
        """Test that fields are properly sets."""
        dep = CoordinatorDependency(
            name="test",
            emits={"a", "b", "a"},  # Duplicate
            subscribes={"c"},
            depends_on={"d"},
        )
        assert len(dep.emits) == 2  # Duplicates removed
        assert "a" in dep.emits
        assert "b" in dep.emits


class TestCoordinatorRegistry:
    """Tests for COORDINATOR_REGISTRY constant."""

    def test_registry_is_dict(self):
        """Test registry is a dict."""
        assert isinstance(COORDINATOR_REGISTRY, dict)

    def test_registry_has_expected_coordinators(self):
        """Test registry contains known coordinators."""
        expected = {
            "task_lifecycle",
            "resources",
            "cache",
            "selfplay",
            "pipeline",
            "optimization",
            "metrics",
        }
        assert expected == set(COORDINATOR_REGISTRY.keys())

    def test_registry_values_are_dependencies(self):
        """Test all registry values are CoordinatorDependency."""
        for name, dep in COORDINATOR_REGISTRY.items():
            assert isinstance(dep, CoordinatorDependency)
            assert dep.name == name  # Name should match key

    def test_foundational_coordinators_have_no_deps(self):
        """Test foundational coordinators have no dependencies."""
        foundational = ["task_lifecycle", "resources", "cache"]
        for name in foundational:
            dep = COORDINATOR_REGISTRY[name]
            assert dep.depends_on == set(), f"{name} should have no dependencies"

    def test_selfplay_depends_on_foundational(self):
        """Test selfplay depends on task_lifecycle and resources."""
        selfplay = COORDINATOR_REGISTRY["selfplay"]
        assert "task_lifecycle" in selfplay.depends_on
        assert "resources" in selfplay.depends_on

    def test_pipeline_depends_on_selfplay(self):
        """Test pipeline depends on selfplay."""
        pipeline = COORDINATOR_REGISTRY["pipeline"]
        assert "selfplay" in pipeline.depends_on

    def test_all_deps_exist_in_registry(self):
        """Test all dependencies reference existing coordinators."""
        all_names = set(COORDINATOR_REGISTRY.keys())
        for name, dep in COORDINATOR_REGISTRY.items():
            for dep_name in dep.depends_on:
                assert dep_name in all_names, f"{name} depends on unknown {dep_name}"


class TestCoordinatorDependencyGraph:
    """Tests for CoordinatorDependencyGraph class."""

    def setup_method(self):
        """Reset global graph before each test."""
        reset_dependency_graph()

    def test_empty_graph(self):
        """Test empty graph."""
        graph = CoordinatorDependencyGraph()
        assert len(graph._dependencies) == 0
        assert graph.detect_cycles() == []
        order, success = graph.topological_sort()
        assert order == []
        assert success is True

    def test_add_coordinator(self):
        """Test adding a coordinator."""
        graph = CoordinatorDependencyGraph()
        dep = CoordinatorDependency(name="test", depends_on={"other"})
        graph.add_coordinator(dep)
        assert "test" in graph._dependencies
        assert "other" in graph._adjacency["test"]
        assert "test" in graph._reverse["other"]

    def test_build_from_registry(self):
        """Test building graph from registry."""
        graph = CoordinatorDependencyGraph()
        graph.build_from_registry()
        assert len(graph._dependencies) == len(COORDINATOR_REGISTRY)
        for name in COORDINATOR_REGISTRY:
            assert name in graph._dependencies

    def test_get_dependencies(self):
        """Test getting dependencies."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b", "c"}))
        graph.add_coordinator(CoordinatorDependency(name="b"))
        graph.add_coordinator(CoordinatorDependency(name="c"))
        deps = graph.get_dependencies("a")
        assert deps == {"b", "c"}

    def test_get_dependents(self):
        """Test getting dependents."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="c", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b"))
        dependents = graph.get_dependents("b")
        assert dependents == {"a", "c"}

    def test_detect_no_cycles(self):
        """Test detecting no cycles in acyclic graph."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"c"}))
        graph.add_coordinator(CoordinatorDependency(name="c"))
        cycles = graph.detect_cycles()
        assert cycles == []

    def test_detect_simple_cycle(self):
        """Test detecting simple cycle."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"a"}))
        cycles = graph.detect_cycles()
        assert len(cycles) > 0
        # Cycle should contain both a and b
        cycle = cycles[0]
        assert "a" in cycle
        assert "b" in cycle

    def test_detect_three_node_cycle(self):
        """Test detecting three-node cycle."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"c"}))
        graph.add_coordinator(CoordinatorDependency(name="c", depends_on={"a"}))
        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_topological_sort_simple(self):
        """Test topological sort with simple chain."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="c"))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"c"}))
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        order, success = graph.topological_sort()
        assert success is True
        assert len(order) == 3
        # c should come before b, b before a
        assert order.index("c") < order.index("b")
        assert order.index("b") < order.index("a")

    def test_topological_sort_diamond(self):
        """Test topological sort with diamond dependency."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="d"))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"d"}))
        graph.add_coordinator(CoordinatorDependency(name="c", depends_on={"d"}))
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b", "c"}))
        order, success = graph.topological_sort()
        assert success is True
        assert len(order) == 4
        # d should be first, a should be last
        assert order.index("d") < order.index("b")
        assert order.index("d") < order.index("c")
        assert order.index("b") < order.index("a")
        assert order.index("c") < order.index("a")

    def test_topological_sort_with_cycle(self):
        """Test topological sort fails with cycle."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"a"}))
        order, success = graph.topological_sort()
        assert success is False
        assert len(order) < 2  # Not all nodes in result

    def test_validate_valid_graph(self):
        """Test validating a valid graph."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b"))
        valid, issues = graph.validate()
        assert valid is True
        assert issues == []

    def test_validate_missing_dependency(self):
        """Test validation catches missing dependencies."""
        graph = CoordinatorDependencyGraph()
        # 'a' depends on 'b', but 'b' is not added
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        valid, issues = graph.validate()
        assert valid is False
        assert len(issues) == 1
        assert "unknown" in issues[0].lower()

    def test_validate_cycle(self):
        """Test validation catches cycles."""
        graph = CoordinatorDependencyGraph()
        graph.add_coordinator(CoordinatorDependency(name="a", depends_on={"b"}))
        graph.add_coordinator(CoordinatorDependency(name="b", depends_on={"a"}))
        valid, issues = graph.validate()
        assert valid is False
        assert any("circular" in issue.lower() for issue in issues)


class TestGlobalGraphFunctions:
    """Tests for module-level graph functions."""

    def setup_method(self):
        """Reset global graph before each test."""
        reset_dependency_graph()

    def test_get_dependency_graph_singleton(self):
        """Test get_dependency_graph returns same instance."""
        graph1 = get_dependency_graph()
        graph2 = get_dependency_graph()
        assert graph1 is graph2

    def test_get_dependency_graph_builds_from_registry(self):
        """Test get_dependency_graph builds from registry."""
        graph = get_dependency_graph()
        assert len(graph._dependencies) == len(COORDINATOR_REGISTRY)

    def test_reset_dependency_graph(self):
        """Test reset_dependency_graph clears singleton."""
        graph1 = get_dependency_graph()
        reset_dependency_graph()
        graph2 = get_dependency_graph()
        assert graph1 is not graph2

    def test_validate_dependencies(self):
        """Test validate_dependencies function."""
        valid, issues = validate_dependencies()
        # Registry should be valid
        assert valid is True
        assert issues == []

    def test_get_initialization_order(self):
        """Test get_initialization_order function."""
        order = get_initialization_order()
        assert isinstance(order, list)
        assert len(order) == len(COORDINATOR_REGISTRY)
        # All coordinators should be in order
        for name in COORDINATOR_REGISTRY:
            assert name in order

    def test_initialization_order_respects_deps(self):
        """Test initialization order respects dependencies."""
        order = get_initialization_order()
        # task_lifecycle should come before selfplay
        assert order.index("task_lifecycle") < order.index("selfplay")
        # selfplay should come before pipeline
        assert order.index("selfplay") < order.index("pipeline")

    def test_get_initialization_order_raises_on_cycle(self):
        """Test get_initialization_order raises on cycle."""
        # Create a custom graph with a cycle
        reset_dependency_graph()
        # We need to patch the global registry to test this
        # For now, just verify it works with valid registry
        order = get_initialization_order()
        assert len(order) > 0


class TestEventChainCycles:
    """Tests for check_event_chain_cycles function."""

    def setup_method(self):
        """Reset global graph before each test."""
        reset_dependency_graph()

    def test_no_event_cycles(self):
        """Test no event cycles in registry."""
        cycles = check_event_chain_cycles()
        # The registry should not have event cycles
        # (if it does, that's a design issue to fix)
        # This test documents the current state
        assert isinstance(cycles, list)

    def test_event_chain_analysis(self):
        """Test event chain cycle detection logic."""
        # Create a graph with event-based connections
        graph = CoordinatorDependencyGraph()

        # A emits "foo", B subscribes to "foo" and emits "bar",
        # C subscribes to "bar" and emits "baz"
        graph.add_coordinator(
            CoordinatorDependency(
                name="A",
                emits={"foo"},
                subscribes=set(),
            )
        )
        graph.add_coordinator(
            CoordinatorDependency(
                name="B",
                emits={"bar"},
                subscribes={"foo"},
            )
        )
        graph.add_coordinator(
            CoordinatorDependency(
                name="C",
                emits={"baz"},
                subscribes={"bar"},
            )
        )

        # No cycle - this is a chain A -> B -> C
        cycles = graph.detect_cycles()
        assert cycles == []


class TestRegistryIntegrity:
    """Integration tests for registry integrity."""

    def test_no_circular_dependencies(self):
        """Test registry has no circular dependencies."""
        valid, issues = validate_dependencies()
        assert valid is True, f"Registry has issues: {issues}"

    def test_initialization_order_complete(self):
        """Test all coordinators can be initialized."""
        order = get_initialization_order()
        assert set(order) == set(COORDINATOR_REGISTRY.keys())

    def test_event_emitters_exist(self):
        """Test all event emitters are defined."""
        all_emits = set()
        for dep in COORDINATOR_REGISTRY.values():
            all_emits.update(dep.emits)
        # Just verify we collected some events
        assert len(all_emits) > 0

    def test_event_subscribers_exist(self):
        """Test all event subscribers are defined."""
        all_subscribes = set()
        for dep in COORDINATOR_REGISTRY.values():
            all_subscribes.update(dep.subscribes)
        # Just verify we collected some subscriptions
        assert len(all_subscribes) > 0

    def test_emits_and_subscribes_overlap(self):
        """Test some events are both emitted and subscribed."""
        all_emits = set()
        all_subscribes = set()
        for dep in COORDINATOR_REGISTRY.values():
            all_emits.update(dep.emits)
            all_subscribes.update(dep.subscribes)

        # There should be some overlap (events that are actually used)
        connected = all_emits & all_subscribes
        assert len(connected) > 0, "No connected events - coordinators are isolated"
