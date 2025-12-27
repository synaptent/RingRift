"""Tests for NodeSelector: Node ranking and selection for job dispatch."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Mock NodeInfo for testing
@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing."""

    node_id: str
    host: str = "localhost"
    port: int = 8770
    has_gpu: bool = True
    gpu_name: str = "RTX 4090"
    memory_gb: float = 24.0
    gpu_percent: float = 50.0
    cpu_count: int = 32
    cpu_percent: float = 25.0
    retired: bool = False
    _alive: bool = True
    _healthy: bool = True
    _gpu_power: float = 100.0
    _cpu_power: float = 50.0
    _load: float = 0.5

    def is_alive(self) -> bool:
        return self._alive

    def is_healthy(self) -> bool:
        return self._healthy

    def gpu_power_score(self) -> float:
        return self._gpu_power if self.has_gpu else 0.0

    def cpu_power_score(self) -> float:
        return self._cpu_power

    def get_load_score(self) -> float:
        return self._load


@dataclass
class MockTrainingJob:
    """Mock training job for testing."""

    job_id: str
    worker_node: str
    status: str = "running"


# Import after mocking
from scripts.p2p.managers.node_selector import NodeSelector, TRAINING_NODE_COUNT


class TestNodeSelectorInitialization:
    """Test NodeSelector initialization."""

    def test_init_with_required_params(self):
        """Test basic initialization with required params."""
        get_peers = MagicMock(return_value={})
        get_self = MagicMock(return_value=MockNodeInfo("self"))

        selector = NodeSelector(get_peers, get_self)

        assert selector._get_peers is get_peers
        assert selector._get_self_info is get_self
        assert selector._peers_lock is None
        assert selector._get_training_jobs is None

    def test_init_with_lock(self):
        """Test initialization with peers lock."""
        get_peers = MagicMock(return_value={})
        get_self = MagicMock(return_value=MockNodeInfo("self"))
        lock = threading.Lock()

        selector = NodeSelector(get_peers, get_self, peers_lock=lock)

        assert selector._peers_lock is lock

    def test_init_with_training_jobs_getter(self):
        """Test initialization with training jobs getter."""
        get_peers = MagicMock(return_value={})
        get_self = MagicMock(return_value=MockNodeInfo("self"))
        get_jobs = MagicMock(return_value={})

        selector = NodeSelector(get_peers, get_self, get_training_jobs=get_jobs)

        assert selector._get_training_jobs is get_jobs


class TestGetAllNodes:
    """Test _get_all_nodes method."""

    def test_get_all_nodes_includes_self(self):
        """Test that _get_all_nodes includes self by default."""
        peer = MockNodeInfo("peer-1", host="10.0.0.1")
        self_node = MockNodeInfo("self", host="10.0.0.2")

        get_peers = MagicMock(return_value={"peer-1": peer})
        get_self = MagicMock(return_value=self_node)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector._get_all_nodes(include_self=True)

        assert len(nodes) == 2
        assert peer in nodes
        assert self_node in nodes

    def test_get_all_nodes_excludes_self(self):
        """Test that _get_all_nodes can exclude self."""
        peer = MockNodeInfo("peer-1", host="10.0.0.1")
        self_node = MockNodeInfo("self", host="10.0.0.2")

        get_peers = MagicMock(return_value={"peer-1": peer})
        get_self = MagicMock(return_value=self_node)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector._get_all_nodes(include_self=False)

        assert len(nodes) == 1
        assert peer in nodes
        assert self_node not in nodes

    def test_get_all_nodes_with_lock(self):
        """Test that _get_all_nodes uses lock when provided."""
        peer = MockNodeInfo("peer-1")
        lock = threading.Lock()

        get_peers = MagicMock(return_value={"peer-1": peer})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self, peers_lock=lock)
        nodes = selector._get_all_nodes(include_self=True)

        assert len(nodes) == 1
        get_peers.assert_called_once()

    def test_get_all_nodes_handles_none_self(self):
        """Test that _get_all_nodes handles None self_info."""
        peer = MockNodeInfo("peer-1")

        get_peers = MagicMock(return_value={"peer-1": peer})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector._get_all_nodes(include_self=True)

        assert len(nodes) == 1
        assert peer in nodes


class TestGetTrainingPrimaryNodes:
    """Test get_training_primary_nodes method."""

    def test_returns_gpu_nodes_sorted_by_power(self):
        """Test that GPU nodes are returned sorted by power."""
        h100 = MockNodeInfo("h100", has_gpu=True, _gpu_power=200.0)
        a10 = MockNodeInfo("a10", has_gpu=True, _gpu_power=50.0)
        gh200 = MockNodeInfo("gh200", has_gpu=True, _gpu_power=150.0)

        get_peers = MagicMock(return_value={
            "h100": h100,
            "a10": a10,
            "gh200": gh200,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_training_primary_nodes(count=3)

        assert len(nodes) == 3
        assert nodes[0] is h100
        assert nodes[1] is gh200
        assert nodes[2] is a10

    def test_filters_out_non_gpu_nodes(self):
        """Test that non-GPU nodes are excluded."""
        gpu_node = MockNodeInfo("gpu", has_gpu=True, _gpu_power=100.0)
        cpu_node = MockNodeInfo("cpu", has_gpu=False, _gpu_power=0.0)

        get_peers = MagicMock(return_value={"gpu": gpu_node, "cpu": cpu_node})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_training_primary_nodes()

        assert len(nodes) == 1
        assert nodes[0] is gpu_node

    def test_filters_out_dead_nodes(self):
        """Test that dead nodes are excluded."""
        alive = MockNodeInfo("alive", has_gpu=True, _alive=True, _gpu_power=100.0)
        dead = MockNodeInfo("dead", has_gpu=True, _alive=False, _gpu_power=200.0)

        get_peers = MagicMock(return_value={"alive": alive, "dead": dead})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_training_primary_nodes()

        assert len(nodes) == 1
        assert nodes[0] is alive

    def test_filters_out_zero_power_nodes(self):
        """Test that nodes with zero power score are excluded."""
        good = MockNodeInfo("good", has_gpu=True, _gpu_power=100.0)
        zero = MockNodeInfo("zero", has_gpu=True, _gpu_power=0.0)

        get_peers = MagicMock(return_value={"good": good, "zero": zero})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_training_primary_nodes()

        assert len(nodes) == 1
        assert nodes[0] is good

    def test_respects_count_parameter(self):
        """Test that count parameter limits results."""
        nodes_dict = {
            f"node-{i}": MockNodeInfo(f"node-{i}", has_gpu=True, _gpu_power=100.0 - i)
            for i in range(10)
        }

        get_peers = MagicMock(return_value=nodes_dict)
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_training_primary_nodes(count=3)

        assert len(nodes) == 3

    def test_uses_default_count(self):
        """Test that default count is TRAINING_NODE_COUNT."""
        nodes_dict = {
            f"node-{i}": MockNodeInfo(f"node-{i}", has_gpu=True, _gpu_power=100.0 - i)
            for i in range(10)
        }

        get_peers = MagicMock(return_value=nodes_dict)
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_training_primary_nodes()

        assert len(nodes) == min(10, TRAINING_NODE_COUNT)


class TestGetTrainingNodesRanked:
    """Test get_training_nodes_ranked method."""

    def test_returns_ranking_dicts(self):
        """Test that ranked nodes include expected fields."""
        node = MockNodeInfo(
            "node-1",
            has_gpu=True,
            gpu_name="RTX 4090",
            memory_gb=24.0,
            gpu_percent=50.0,
            _gpu_power=100.0,
            _alive=True,
            _healthy=True,
        )

        get_peers = MagicMock(return_value={"node-1": node})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_training_nodes_ranked()

        assert len(result) == 1
        assert result[0]["node_id"] == "node-1"
        assert result[0]["gpu_name"] == "RTX 4090"
        assert result[0]["gpu_power_score"] == 100.0
        assert result[0]["memory_gb"] == 24.0
        assert result[0]["is_alive"] is True
        assert result[0]["is_healthy"] is True
        assert result[0]["gpu_percent"] == 50.0

    def test_sorted_by_power_score(self):
        """Test that results are sorted by power score descending."""
        low = MockNodeInfo("low", has_gpu=True, _gpu_power=50.0)
        high = MockNodeInfo("high", has_gpu=True, _gpu_power=200.0)
        mid = MockNodeInfo("mid", has_gpu=True, _gpu_power=100.0)

        get_peers = MagicMock(return_value={"low": low, "high": high, "mid": mid})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_training_nodes_ranked()

        assert len(result) == 3
        assert result[0]["node_id"] == "high"
        assert result[1]["node_id"] == "mid"
        assert result[2]["node_id"] == "low"


class TestGetBestGPUNodeForTraining:
    """Test get_best_gpu_node_for_training method."""

    def test_returns_best_gpu_node(self):
        """Test that best GPU node is returned."""
        h100 = MockNodeInfo("h100", has_gpu=True, _gpu_power=200.0, _load=0.1)
        a10 = MockNodeInfo("a10", has_gpu=True, _gpu_power=50.0, _load=0.1)

        get_peers = MagicMock(return_value={"h100": h100, "a10": a10})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_gpu_node_for_training()

        assert result is h100

    def test_excludes_retired_nodes(self):
        """Test that retired nodes are excluded."""
        retired = MockNodeInfo("retired", has_gpu=True, _gpu_power=200.0, retired=True)
        active = MockNodeInfo("active", has_gpu=True, _gpu_power=100.0, retired=False)

        get_peers = MagicMock(return_value={"retired": retired, "active": active})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_gpu_node_for_training()

        assert result is active

    def test_excludes_unhealthy_nodes(self):
        """Test that unhealthy nodes are excluded."""
        unhealthy = MockNodeInfo("unhealthy", has_gpu=True, _gpu_power=200.0, _healthy=False)
        healthy = MockNodeInfo("healthy", has_gpu=True, _gpu_power=100.0, _healthy=True)

        get_peers = MagicMock(return_value={"unhealthy": unhealthy, "healthy": healthy})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_gpu_node_for_training()

        assert result is healthy

    def test_exclude_node_ids(self):
        """Test that specified node IDs are excluded."""
        best = MockNodeInfo("best", has_gpu=True, _gpu_power=200.0)
        second = MockNodeInfo("second", has_gpu=True, _gpu_power=100.0)

        get_peers = MagicMock(return_value={"best": best, "second": second})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_gpu_node_for_training(exclude_node_ids={"best"})

        assert result is second

    def test_exclude_nodes_with_training_jobs(self):
        """Test that nodes with active training jobs are excluded."""
        best = MockNodeInfo("best", has_gpu=True, _gpu_power=200.0)
        second = MockNodeInfo("second", has_gpu=True, _gpu_power=100.0)
        training_job = MockTrainingJob("job-1", "best", status="running")

        get_peers = MagicMock(return_value={"best": best, "second": second})
        get_self = MagicMock(return_value=None)
        get_jobs = MagicMock(return_value={"job-1": training_job})

        selector = NodeSelector(get_peers, get_self, get_training_jobs=get_jobs)
        result = selector.get_best_gpu_node_for_training()

        assert result is second

    def test_returns_none_when_no_gpu_nodes(self):
        """Test that None is returned when no GPU nodes available."""
        cpu_node = MockNodeInfo("cpu", has_gpu=False)

        get_peers = MagicMock(return_value={"cpu": cpu_node})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_gpu_node_for_training()

        assert result is None

    def test_prefers_lower_load(self):
        """Test that lower load is preferred when power is equal."""
        busy = MockNodeInfo("busy", has_gpu=True, _gpu_power=100.0, _load=0.9)
        idle = MockNodeInfo("idle", has_gpu=True, _gpu_power=100.0, _load=0.1)

        get_peers = MagicMock(return_value={"busy": busy, "idle": idle})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_gpu_node_for_training()

        assert result is idle

    def test_falls_back_to_busy_nodes_when_all_excluded(self):
        """Test fallback when all available nodes are excluded."""
        best = MockNodeInfo("best", has_gpu=True, _gpu_power=200.0)

        get_peers = MagicMock(return_value={"best": best})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        # Exclude the only node
        result = selector.get_best_gpu_node_for_training(exclude_node_ids={"best"})

        # Should fall back to best since no others available
        assert result is best


class TestGetCPUPrimaryNodes:
    """Test get_cpu_primary_nodes method."""

    def test_returns_nodes_sorted_by_cpu_power(self):
        """Test that nodes are sorted by CPU power."""
        vast_high = MockNodeInfo("vast-1", cpu_count=256, _cpu_power=200.0, _load=0.1)
        lambda_mid = MockNodeInfo("lambda-1", cpu_count=64, _cpu_power=100.0, _load=0.1)
        small = MockNodeInfo("small", cpu_count=8, _cpu_power=25.0, _load=0.1)

        get_peers = MagicMock(return_value={
            "vast-1": vast_high,
            "lambda-1": lambda_mid,
            "small": small,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_cpu_primary_nodes(count=3)

        assert len(nodes) == 3
        assert nodes[0] is vast_high
        assert nodes[1] is lambda_mid
        assert nodes[2] is small

    def test_filters_dead_and_unhealthy(self):
        """Test that dead and unhealthy nodes are filtered."""
        alive = MockNodeInfo("alive", _alive=True, _healthy=True, _cpu_power=50.0)
        dead = MockNodeInfo("dead", _alive=False, _healthy=True, _cpu_power=100.0)
        unhealthy = MockNodeInfo("sick", _alive=True, _healthy=False, _cpu_power=100.0)

        get_peers = MagicMock(return_value={
            "alive": alive,
            "dead": dead,
            "sick": unhealthy,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        nodes = selector.get_cpu_primary_nodes()

        assert len(nodes) == 1
        assert nodes[0] is alive


class TestGetCPUNodesRanked:
    """Test get_cpu_nodes_ranked method."""

    def test_returns_ranking_dicts(self):
        """Test that ranked nodes include expected fields."""
        node = MockNodeInfo(
            "node-1",
            has_gpu=True,
            cpu_count=64,
            cpu_percent=25.0,
            memory_gb=128.0,
            _cpu_power=100.0,
            _alive=True,
            _healthy=True,
        )

        get_peers = MagicMock(return_value={"node-1": node})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_cpu_nodes_ranked()

        assert len(result) == 1
        assert result[0]["node_id"] == "node-1"
        assert result[0]["cpu_count"] == 64
        assert result[0]["cpu_power_score"] == 100.0
        assert result[0]["cpu_percent"] == 25.0
        assert result[0]["memory_gb"] == 128.0
        assert result[0]["is_alive"] is True
        assert result[0]["is_healthy"] is True
        assert result[0]["has_gpu"] is True

    def test_excludes_nodes_with_no_cpu_info(self):
        """Test that nodes without CPU info are excluded."""
        with_cpu = MockNodeInfo("with", cpu_count=64, _cpu_power=100.0)
        no_cpu = MockNodeInfo("without", cpu_count=0, _cpu_power=0.0)

        get_peers = MagicMock(return_value={"with": with_cpu, "without": no_cpu})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_cpu_nodes_ranked()

        assert len(result) == 1
        assert result[0]["node_id"] == "with"


class TestGetBestCPUNodeForGauntlet:
    """Test get_best_cpu_node_for_gauntlet method."""

    def test_prefers_vast_nodes(self):
        """Test that Vast nodes are preferred for gauntlets."""
        vast = MockNodeInfo("vast-12345", cpu_count=256, _cpu_power=200.0)
        lambda_node = MockNodeInfo("lambda-1", cpu_count=64, _cpu_power=150.0)

        get_peers = MagicMock(return_value={
            "vast-12345": vast,
            "lambda-1": lambda_node,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_cpu_node_for_gauntlet()

        assert result is vast

    def test_prefers_high_cpu_count(self):
        """Test that high CPU count nodes are preferred."""
        high_cpu = MockNodeInfo("high", cpu_count=256, _cpu_power=200.0)
        low_cpu = MockNodeInfo("low", cpu_count=8, _cpu_power=25.0)

        get_peers = MagicMock(return_value={"high": high_cpu, "low": low_cpu})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_cpu_node_for_gauntlet()

        assert result is high_cpu

    def test_excludes_retired_nodes(self):
        """Test that retired nodes are excluded."""
        retired = MockNodeInfo("vast-retired", cpu_count=256, _cpu_power=200.0, retired=True)
        active = MockNodeInfo("active", cpu_count=64, _cpu_power=100.0, retired=False)

        get_peers = MagicMock(return_value={"vast-retired": retired, "active": active})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_cpu_node_for_gauntlet()

        assert result is active

    def test_falls_back_to_non_vast_when_no_vast(self):
        """Test fallback to non-Vast nodes when no Vast available."""
        lambda_node = MockNodeInfo("lambda-1", cpu_count=32, _cpu_power=75.0)

        get_peers = MagicMock(return_value={"lambda-1": lambda_node})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_cpu_node_for_gauntlet()

        assert result is lambda_node

    def test_returns_none_when_no_cpu_nodes(self):
        """Test that None is returned when no CPU nodes."""
        get_peers = MagicMock(return_value={})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_best_cpu_node_for_gauntlet()

        assert result is None


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_alive_gpu_nodes(self):
        """Test get_alive_gpu_nodes returns alive GPU nodes."""
        alive_gpu = MockNodeInfo("alive-gpu", has_gpu=True, _alive=True)
        dead_gpu = MockNodeInfo("dead-gpu", has_gpu=True, _alive=False)
        alive_cpu = MockNodeInfo("alive-cpu", has_gpu=False, _alive=True)

        get_peers = MagicMock(return_value={
            "alive-gpu": alive_gpu,
            "dead-gpu": dead_gpu,
            "alive-cpu": alive_cpu,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_alive_gpu_nodes()

        assert len(result) == 1
        assert result[0] is alive_gpu

    def test_get_alive_nodes(self):
        """Test get_alive_nodes returns all alive nodes."""
        alive1 = MockNodeInfo("alive1", _alive=True)
        alive2 = MockNodeInfo("alive2", _alive=True)
        dead = MockNodeInfo("dead", _alive=False)

        get_peers = MagicMock(return_value={
            "alive1": alive1,
            "alive2": alive2,
            "dead": dead,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_alive_nodes()

        assert len(result) == 2
        assert alive1 in result
        assert alive2 in result

    def test_get_healthy_nodes(self):
        """Test get_healthy_nodes returns healthy nodes."""
        healthy = MockNodeInfo("healthy", _healthy=True)
        unhealthy = MockNodeInfo("unhealthy", _healthy=False)

        get_peers = MagicMock(return_value={
            "healthy": healthy,
            "unhealthy": unhealthy,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.get_healthy_nodes()

        assert len(result) == 1
        assert result[0] is healthy

    def test_count_alive_peers(self):
        """Test count_alive_peers counts alive peers correctly."""
        alive1 = MockNodeInfo("alive1", _alive=True)
        alive2 = MockNodeInfo("alive2", _alive=True)
        dead = MockNodeInfo("dead", _alive=False)

        get_peers = MagicMock(return_value={
            "alive1": alive1,
            "alive2": alive2,
            "dead": dead,
        })
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self)
        result = selector.count_alive_peers()

        assert result == 2

    def test_count_alive_peers_with_lock(self):
        """Test count_alive_peers uses lock when provided."""
        alive = MockNodeInfo("alive", _alive=True)
        lock = threading.Lock()

        get_peers = MagicMock(return_value={"alive": alive})
        get_self = MagicMock(return_value=None)

        selector = NodeSelector(get_peers, get_self, peers_lock=lock)
        result = selector.count_alive_peers()

        assert result == 1
        get_peers.assert_called()


class TestThreadSafety:
    """Test thread safety of NodeSelector."""

    def test_concurrent_access(self):
        """Test that concurrent access with lock is safe."""
        peers = {
            f"node-{i}": MockNodeInfo(f"node-{i}", has_gpu=True, _gpu_power=100.0)
            for i in range(10)
        }
        lock = threading.RLock()

        get_peers = MagicMock(return_value=peers)
        get_self = MagicMock(return_value=MockNodeInfo("self"))

        selector = NodeSelector(get_peers, get_self, peers_lock=lock)

        results = []

        def reader():
            for _ in range(100):
                nodes = selector.get_training_primary_nodes()
                results.append(len(nodes))

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 500
        assert all(r == min(11, TRAINING_NODE_COUNT) for r in results)  # 10 peers + 1 self
