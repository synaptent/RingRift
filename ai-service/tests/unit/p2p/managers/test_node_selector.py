"""Unit tests for NodeSelector.

Tests node ranking, selection, health management, and event subscription.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from scripts.p2p.managers.node_selector import NodeSelector, TRAINING_NODE_COUNT


class MockNodeInfo:
    """Mock NodeInfo for testing."""

    def __init__(
        self,
        node_id: str,
        has_gpu: bool = False,
        gpu_name: str = "",
        gpu_power: float = 0.0,
        cpu_count: int = 4,
        cpu_power: float = 0.0,
        memory_gb: float = 16.0,
        gpu_percent: float = 0.0,
        cpu_percent: float = 0.0,
        is_alive_val: bool = True,
        is_healthy_val: bool = True,
        retired: bool = False,
        capabilities: list[str] | None = None,
        load_score: float = 0.0,
    ):
        self.node_id = node_id
        self.has_gpu = has_gpu
        self.gpu_name = gpu_name
        self._gpu_power = gpu_power
        self.cpu_count = cpu_count
        self._cpu_power = cpu_power
        self.memory_gb = memory_gb
        self.gpu_percent = gpu_percent
        self.cpu_percent = cpu_percent
        self._is_alive = is_alive_val
        self._is_healthy = is_healthy_val
        self.retired = retired
        self.capabilities = capabilities or []
        self._load_score = load_score

    def is_alive(self) -> bool:
        return self._is_alive

    def is_healthy(self) -> bool:
        return self._is_healthy

    def gpu_power_score(self) -> float:
        return self._gpu_power

    def cpu_power_score(self) -> float:
        return self._cpu_power

    def get_load_score(self) -> float:
        return self._load_score


class TestNodeSelectorInit:
    """Tests for NodeSelector initialization."""

    def test_init_with_minimal_args(self):
        """Test initialization with only required arguments."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )
        assert selector._get_peers() == {}
        assert selector._get_self_info() is None
        assert selector._peers_lock is None
        assert selector._get_training_jobs is None
        assert selector._unhealthy_nodes == set()
        assert selector._subscribed is False

    def test_init_with_all_args(self):
        """Test initialization with all arguments."""
        lock = threading.Lock()
        peers = {"node1": MockNodeInfo("node1")}
        self_info = MockNodeInfo("self")
        training_jobs = {"job1": MagicMock()}

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: self_info,
            peers_lock=lock,
            get_training_jobs=lambda: training_jobs,
        )

        assert selector._get_peers() == peers
        assert selector._get_self_info() == self_info
        assert selector._peers_lock is lock
        assert selector._get_training_jobs() == training_jobs


class TestGetAllNodes:
    """Tests for _get_all_nodes helper."""

    def test_get_all_nodes_with_self(self):
        """Test getting all nodes including self."""
        peers = {
            "node1": MockNodeInfo("node1"),
            "node2": MockNodeInfo("node2"),
        }
        self_info = MockNodeInfo("self")

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: self_info,
        )

        nodes = selector._get_all_nodes(include_self=True)
        node_ids = {n.node_id for n in nodes}
        assert node_ids == {"node1", "node2", "self"}

    def test_get_all_nodes_without_self(self):
        """Test getting all nodes excluding self."""
        peers = {
            "node1": MockNodeInfo("node1"),
            "node2": MockNodeInfo("node2"),
        }
        self_info = MockNodeInfo("self")

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: self_info,
        )

        nodes = selector._get_all_nodes(include_self=False)
        node_ids = {n.node_id for n in nodes}
        assert node_ids == {"node1", "node2"}

    def test_get_all_nodes_with_lock(self):
        """Test getting all nodes with thread lock."""
        lock = threading.Lock()
        peers = {"node1": MockNodeInfo("node1")}

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
            peers_lock=lock,
        )

        nodes = selector._get_all_nodes(include_self=False)
        assert len(nodes) == 1


class TestGPUNodeSelection:
    """Tests for GPU node selection methods."""

    def test_get_training_primary_nodes_returns_gpu_nodes(self):
        """Test that only GPU nodes with training capability are returned."""
        peers = {
            "gpu1": MockNodeInfo(
                "gpu1", has_gpu=True, gpu_power=100.0,
                capabilities=["training", "selfplay"]
            ),
            "gpu2": MockNodeInfo(
                "gpu2", has_gpu=True, gpu_power=80.0,
                capabilities=["training"]
            ),
            "cpu1": MockNodeInfo(
                "cpu1", has_gpu=False, cpu_power=50.0,
                capabilities=["training"]
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_training_primary_nodes(count=5)
        node_ids = [n.node_id for n in nodes]

        assert "gpu1" in node_ids
        assert "gpu2" in node_ids
        assert "cpu1" not in node_ids

    def test_get_training_primary_nodes_sorted_by_gpu_power(self):
        """Test that nodes are sorted by GPU power (highest first)."""
        peers = {
            "gpu_low": MockNodeInfo(
                "gpu_low", has_gpu=True, gpu_power=50.0,
                capabilities=["training"]
            ),
            "gpu_high": MockNodeInfo(
                "gpu_high", has_gpu=True, gpu_power=200.0,
                capabilities=["training"]
            ),
            "gpu_mid": MockNodeInfo(
                "gpu_mid", has_gpu=True, gpu_power=100.0,
                capabilities=["training"]
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_training_primary_nodes(count=3)
        powers = [n.gpu_power_score() for n in nodes]

        assert powers == sorted(powers, reverse=True)
        assert nodes[0].node_id == "gpu_high"
        assert nodes[1].node_id == "gpu_mid"
        assert nodes[2].node_id == "gpu_low"

    def test_get_training_primary_nodes_respects_count(self):
        """Test that only 'count' nodes are returned."""
        peers = {
            f"gpu{i}": MockNodeInfo(
                f"gpu{i}", has_gpu=True, gpu_power=float(i * 10),
                capabilities=["training"]
            )
            for i in range(10)
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_training_primary_nodes(count=3)
        assert len(nodes) == 3

    def test_get_training_primary_nodes_filters_dead_nodes(self):
        """Test that dead nodes are excluded."""
        peers = {
            "alive": MockNodeInfo(
                "alive", has_gpu=True, gpu_power=100.0,
                is_alive_val=True, capabilities=["training"]
            ),
            "dead": MockNodeInfo(
                "dead", has_gpu=True, gpu_power=200.0,
                is_alive_val=False, capabilities=["training"]
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_training_primary_nodes(count=5)
        node_ids = [n.node_id for n in nodes]

        assert "alive" in node_ids
        assert "dead" not in node_ids

    def test_get_best_gpu_node_for_training(self):
        """Test getting the best GPU node for training."""
        peers = {
            "h100": MockNodeInfo(
                "h100", has_gpu=True, gpu_power=300.0,
                capabilities=["training"]
            ),
            "a10": MockNodeInfo(
                "a10", has_gpu=True, gpu_power=100.0,
                capabilities=["training"]
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        best = selector.get_best_gpu_node_for_training()
        assert best is not None
        assert best.node_id == "h100"

    def test_get_best_gpu_node_excludes_specified_nodes(self):
        """Test that excluded nodes are skipped."""
        peers = {
            "h100": MockNodeInfo(
                "h100", has_gpu=True, gpu_power=300.0,
                capabilities=["training"]
            ),
            "a10": MockNodeInfo(
                "a10", has_gpu=True, gpu_power=100.0,
                capabilities=["training"]
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        best = selector.get_best_gpu_node_for_training(exclude_node_ids={"h100"})
        assert best is not None
        assert best.node_id == "a10"

    def test_get_best_gpu_node_excludes_running_training(self):
        """Test that nodes with running training are excluded."""
        job = MagicMock()
        job.worker_node = "h100"
        job.status = "running"

        peers = {
            "h100": MockNodeInfo(
                "h100", has_gpu=True, gpu_power=300.0,
                capabilities=["training"]
            ),
            "a10": MockNodeInfo(
                "a10", has_gpu=True, gpu_power=100.0,
                capabilities=["training"]
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
            get_training_jobs=lambda: {"job1": job},
        )

        best = selector.get_best_gpu_node_for_training()
        assert best is not None
        assert best.node_id == "a10"

    def test_get_best_gpu_node_returns_none_when_no_nodes(self):
        """Test that None is returned when no suitable nodes exist."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        best = selector.get_best_gpu_node_for_training()
        assert best is None


class TestCPUNodeSelection:
    """Tests for CPU node selection methods."""

    def test_get_cpu_primary_nodes(self):
        """Test getting CPU primary nodes."""
        peers = {
            "vast_high": MockNodeInfo(
                "vast_high", cpu_count=200, cpu_power=400.0
            ),
            "nebius": MockNodeInfo(
                "nebius", cpu_count=64, cpu_power=128.0
            ),
            "small": MockNodeInfo(
                "small", cpu_count=4, cpu_power=8.0
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_cpu_primary_nodes(count=2)
        assert len(nodes) == 2
        assert nodes[0].node_id == "vast_high"
        assert nodes[1].node_id == "nebius"

    def test_get_best_cpu_node_for_gauntlet_prefers_vast(self):
        """Test that Vast nodes are preferred for gauntlet."""
        peers = {
            "vast-12345": MockNodeInfo(
                "vast-12345", cpu_count=200, cpu_power=400.0
            ),
            "nebius": MockNodeInfo(
                "nebius", cpu_count=256, cpu_power=512.0  # Higher power
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        best = selector.get_best_cpu_node_for_gauntlet()
        # Vast node preferred due to name containing "vast"
        assert best is not None
        # Both should qualify since nebius has 256 >= 64 CPUs
        assert best.node_id in ("vast-12345", "nebius")

    def test_get_best_cpu_node_for_gauntlet_filters_unhealthy(self):
        """Test that unhealthy nodes are excluded."""
        peers = {
            "healthy": MockNodeInfo(
                "healthy", cpu_count=100, cpu_power=200.0,
                is_healthy_val=True
            ),
            "unhealthy": MockNodeInfo(
                "unhealthy", cpu_count=200, cpu_power=400.0,
                is_healthy_val=False
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        best = selector.get_best_cpu_node_for_gauntlet()
        assert best is not None
        assert best.node_id == "healthy"


class TestEngineModeSelection:
    """Tests for GPU-aware engine mode selection."""

    def test_engine_mode_requires_gpu_for_neural_modes(self):
        """Test that neural network modes require GPU."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        gpu_modes = ["gumbel-mcts", "mcts", "nnue-guided", "policy-only", "gmo"]
        for mode in gpu_modes:
            assert selector.engine_mode_requires_gpu(mode) is True, f"{mode} should require GPU"

    def test_engine_mode_does_not_require_gpu_for_cpu_modes(self):
        """Test that CPU modes don't require GPU."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        cpu_modes = ["heuristic-only", "heuristic", "random", "maxn", "brs"]
        for mode in cpu_modes:
            assert selector.engine_mode_requires_gpu(mode) is False, f"{mode} should not require GPU"

    def test_get_nodes_for_engine_mode_gpu_mode(self):
        """Test getting nodes for GPU engine mode."""
        peers = {
            "gpu1": MockNodeInfo("gpu1", has_gpu=True, gpu_power=100.0),
            "cpu1": MockNodeInfo("cpu1", has_gpu=False, cpu_power=50.0),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_nodes_for_engine_mode("gumbel-mcts")
        node_ids = [n.node_id for n in nodes]

        assert "gpu1" in node_ids
        assert "cpu1" not in node_ids

    def test_get_nodes_for_engine_mode_cpu_mode(self):
        """Test getting nodes for CPU engine mode."""
        peers = {
            "gpu1": MockNodeInfo("gpu1", has_gpu=True, gpu_power=100.0),
            "cpu1": MockNodeInfo("cpu1", has_gpu=False, cpu_power=50.0),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_nodes_for_engine_mode("heuristic")
        node_ids = [n.node_id for n in nodes]

        assert "gpu1" in node_ids
        assert "cpu1" in node_ids

    def test_get_nodes_for_engine_mode_excludes_unhealthy_list(self):
        """Test that nodes in _unhealthy_nodes are excluded."""
        peers = {
            "healthy": MockNodeInfo("healthy", has_gpu=True, gpu_power=100.0),
            "marked_bad": MockNodeInfo("marked_bad", has_gpu=True, gpu_power=200.0),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("marked_bad")

        nodes = selector.get_nodes_for_engine_mode("gumbel-mcts")
        node_ids = [n.node_id for n in nodes]

        assert "healthy" in node_ids
        assert "marked_bad" not in node_ids

    def test_get_gpu_nodes_for_selfplay(self):
        """Test getting GPU nodes for selfplay."""
        peers = {
            "h100": MockNodeInfo("h100", has_gpu=True, gpu_power=300.0, load_score=0.2),
            "a10": MockNodeInfo("a10", has_gpu=True, gpu_power=100.0, load_score=0.1),
            "cpu": MockNodeInfo("cpu", has_gpu=False, cpu_power=50.0),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_gpu_nodes_for_selfplay(count=2)
        node_ids = [n.node_id for n in nodes]

        assert len(nodes) == 2
        assert "cpu" not in node_ids

    def test_get_best_node_for_selfplay_gpu_mode(self):
        """Test getting best node for GPU selfplay."""
        peers = {
            "h100": MockNodeInfo("h100", has_gpu=True, gpu_power=300.0),
            "a10": MockNodeInfo("a10", has_gpu=True, gpu_power=100.0),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        best = selector.get_best_node_for_selfplay("gumbel-mcts")
        assert best is not None
        assert best.node_id == "h100"

    def test_get_best_node_for_selfplay_cpu_mode(self):
        """Test getting best node for CPU selfplay."""
        peers = {
            "high_cpu": MockNodeInfo("high_cpu", cpu_power=200.0),
            "low_cpu": MockNodeInfo("low_cpu", cpu_power=50.0),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        best = selector.get_best_node_for_selfplay("heuristic")
        assert best is not None
        assert best.node_id == "high_cpu"


class TestHealthStateManagement:
    """Tests for health state management."""

    def test_mark_node_unhealthy(self):
        """Test marking a node as unhealthy."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        selector.mark_node_unhealthy("node1", "connection timeout")

        assert "node1" in selector._unhealthy_nodes
        assert selector._unhealthy_reasons["node1"] == "connection timeout"

    def test_mark_node_unhealthy_schedules_probe(self):
        """Test that marking unhealthy schedules a probe by default."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        selector.mark_node_unhealthy("node1", "timeout", schedule_probe=True)
        assert "node1" in selector._pending_probes

    def test_mark_node_unhealthy_no_probe(self):
        """Test marking unhealthy without scheduling probe."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        selector.mark_node_unhealthy("node1", "timeout", schedule_probe=False)
        assert "node1" not in selector._pending_probes

    def test_mark_node_healthy(self):
        """Test marking a node as healthy."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        selector._unhealthy_nodes.add("node1")
        selector._unhealthy_reasons["node1"] = "was bad"

        selector.mark_node_healthy("node1")

        assert "node1" not in selector._unhealthy_nodes
        assert "node1" not in selector._unhealthy_reasons

    def test_is_node_healthy(self):
        """Test checking node health status."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        assert selector.is_node_healthy("node1") is True

        selector._unhealthy_nodes.add("node1")
        assert selector.is_node_healthy("node1") is False

    def test_get_unhealthy_nodes(self):
        """Test getting all unhealthy nodes."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        selector._unhealthy_nodes.add("node1")
        selector._unhealthy_nodes.add("node2")
        selector._unhealthy_reasons["node1"] = "timeout"

        unhealthy = selector.get_unhealthy_nodes()

        assert "node1" in unhealthy
        assert "node2" in unhealthy
        assert unhealthy["node1"] == "timeout"
        assert unhealthy["node2"] == ""


class TestUnhealthyNodeRecovery:
    """Tests for unhealthy node recovery."""

    def test_recover_unhealthy_nodes_recovers_healthy(self):
        """Test that healthy nodes are recovered."""
        peers = {
            "node1": MockNodeInfo("node1", is_alive_val=True, is_healthy_val=True),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("node1")
        selector._unhealthy_reasons["node1"] = "was bad"

        recovered = selector.recover_unhealthy_nodes()

        assert "node1" in recovered
        assert "node1" not in selector._unhealthy_nodes

    def test_recover_unhealthy_nodes_keeps_still_unhealthy(self):
        """Test that still-unhealthy nodes are not recovered."""
        peers = {
            "node1": MockNodeInfo("node1", is_alive_val=False, is_healthy_val=False),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("node1")

        recovered = selector.recover_unhealthy_nodes()

        assert "node1" not in recovered
        assert "node1" in selector._unhealthy_nodes

    def test_recover_unhealthy_nodes_empty_set(self):
        """Test recovery with no unhealthy nodes."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        recovered = selector.recover_unhealthy_nodes()
        assert recovered == []

    def test_recover_specific_nodes(self):
        """Test recovering specific nodes."""
        peers = {
            "node1": MockNodeInfo("node1", is_alive_val=True, is_healthy_val=True),
            "node2": MockNodeInfo("node2", is_alive_val=False, is_healthy_val=False),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("node1")
        selector._unhealthy_nodes.add("node2")

        recovered = selector._recover_specific_nodes({"node1", "node2"})

        assert "node1" in recovered
        assert "node2" not in recovered

    def test_get_recovery_candidates(self):
        """Test getting recovery candidate info."""
        peers = {
            "recoverable": MockNodeInfo(
                "recoverable", is_alive_val=True, is_healthy_val=True
            ),
            "not_recoverable": MockNodeInfo(
                "not_recoverable", is_alive_val=False, is_healthy_val=False
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("recoverable")
        selector._unhealthy_nodes.add("not_recoverable")
        selector._unhealthy_reasons["recoverable"] = "timeout"

        candidates = selector.get_recovery_candidates()

        assert len(candidates) == 2

        recoverable = next(c for c in candidates if c["node_id"] == "recoverable")
        assert recoverable["can_recover"] is True
        assert recoverable["reason"] == "timeout"

        not_recoverable = next(c for c in candidates if c["node_id"] == "not_recoverable")
        assert not_recoverable["can_recover"] is False


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_healthy(self):
        """Test health check with healthy cluster."""
        peers = {
            "gpu1": MockNodeInfo("gpu1", has_gpu=True, is_alive_val=True),
            "gpu2": MockNodeInfo("gpu2", has_gpu=True, is_alive_val=True),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        result = selector.health_check()

        assert result.healthy is True
        assert result.details["alive_nodes"] == 2
        assert result.details["gpu_nodes"] == 2

    def test_health_check_no_alive_nodes(self):
        """Test health check with no alive nodes."""
        peers = {
            "dead1": MockNodeInfo("dead1", is_alive_val=False),
            "dead2": MockNodeInfo("dead2", is_alive_val=False),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        result = selector.health_check()

        assert result.healthy is False
        assert "No alive nodes" in result.message

    def test_health_check_no_gpu_nodes(self):
        """Test health check with no GPU nodes."""
        peers = {
            "cpu1": MockNodeInfo("cpu1", has_gpu=False, is_alive_val=True),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        result = selector.health_check()

        # Should be degraded but not error
        assert result.details["gpu_nodes"] == 0
        assert "No GPU nodes" in (result.message or "")

    def test_health_check_with_unhealthy_nodes(self):
        """Test health check reports unhealthy nodes."""
        peers = {
            "node1": MockNodeInfo("node1", is_alive_val=True),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("bad_node")
        selector._unhealthy_reasons["bad_node"] = "connection failed"

        result = selector.health_check()

        assert result.details["errors_count"] == 1
        assert "bad_node" in result.details["unhealthy_node_ids"]


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_alive_gpu_nodes(self):
        """Test getting alive GPU nodes."""
        peers = {
            "gpu_alive": MockNodeInfo("gpu_alive", has_gpu=True, is_alive_val=True),
            "gpu_dead": MockNodeInfo("gpu_dead", has_gpu=True, is_alive_val=False),
            "cpu_alive": MockNodeInfo("cpu_alive", has_gpu=False, is_alive_val=True),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_alive_gpu_nodes()
        node_ids = [n.node_id for n in nodes]

        assert "gpu_alive" in node_ids
        assert "gpu_dead" not in node_ids
        assert "cpu_alive" not in node_ids

    def test_get_alive_nodes(self):
        """Test getting all alive nodes."""
        peers = {
            "alive1": MockNodeInfo("alive1", is_alive_val=True),
            "alive2": MockNodeInfo("alive2", is_alive_val=True),
            "dead": MockNodeInfo("dead", is_alive_val=False),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_alive_nodes()
        assert len(nodes) == 2

    def test_get_healthy_nodes(self):
        """Test getting all healthy nodes."""
        peers = {
            "healthy1": MockNodeInfo("healthy1", is_healthy_val=True),
            "unhealthy": MockNodeInfo("unhealthy", is_healthy_val=False),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        nodes = selector.get_healthy_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "healthy1"

    def test_count_alive_peers(self):
        """Test counting alive peers."""
        peers = {
            "alive1": MockNodeInfo("alive1", is_alive_val=True),
            "alive2": MockNodeInfo("alive2", is_alive_val=True),
            "dead": MockNodeInfo("dead", is_alive_val=False),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        count = selector.count_alive_peers()
        assert count == 2

    def test_get_training_nodes_ranked(self):
        """Test getting ranked training nodes for dashboard."""
        peers = {
            "h100": MockNodeInfo(
                "h100", has_gpu=True, gpu_name="H100",
                gpu_power=300.0, memory_gb=80.0
            ),
            "a10": MockNodeInfo(
                "a10", has_gpu=True, gpu_name="A10",
                gpu_power=100.0, memory_gb=24.0
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        ranked = selector.get_training_nodes_ranked()

        assert len(ranked) == 2
        assert ranked[0]["node_id"] == "h100"
        assert ranked[0]["gpu_power_score"] == 300.0
        assert ranked[1]["node_id"] == "a10"

    def test_get_cpu_nodes_ranked(self):
        """Test getting ranked CPU nodes for dashboard."""
        peers = {
            "vast": MockNodeInfo(
                "vast", cpu_count=200, cpu_power=400.0
            ),
            "small": MockNodeInfo(
                "small", cpu_count=4, cpu_power=8.0
            ),
        }

        selector = NodeSelector(
            get_peers=lambda: peers,
            get_self_info=lambda: None,
        )

        ranked = selector.get_cpu_nodes_ranked()

        assert len(ranked) == 2
        assert ranked[0]["node_id"] == "vast"
        assert ranked[0]["cpu_power_score"] == 400.0


class TestAsyncLifecycle:
    """Tests for async start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_recovery_task(self):
        """Test that start() creates the recovery task."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        await selector.start()

        assert selector._running is True
        assert selector._recovery_task is not None

        # Stop with error handling for CancelledError
        try:
            await selector.stop()
        except asyncio.CancelledError:
            pass  # Expected when stopping background task

    @pytest.mark.asyncio
    async def test_stop_cancels_recovery_task(self):
        """Test that stop() cancels the recovery task."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        await selector.start()

        try:
            await selector.stop()
        except asyncio.CancelledError:
            pass  # Expected when stopping background task

        assert selector._running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Test that calling start() twice doesn't create duplicate tasks."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        await selector.start()
        task1 = selector._recovery_task

        await selector.start()  # Second call
        task2 = selector._recovery_task

        assert task1 is task2

        try:
            await selector.stop()
        except asyncio.CancelledError:
            pass  # Expected when stopping background task


class TestEventSubscription:
    """Tests for event subscription."""

    def test_subscribe_to_events_sets_flag(self):
        """Test that subscription sets the _subscribed flag."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        # Mock the event bus
        with patch("scripts.p2p.managers.node_selector.NodeSelector.subscribe_to_events") as mock:
            mock.return_value = None
            selector._subscribed = True

        assert selector._subscribed is True

    def test_subscribe_is_idempotent(self):
        """Test that subscribe_to_events is idempotent."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )
        selector._subscribed = True

        # Should return early without trying to subscribe again
        selector.subscribe_to_events()
        assert selector._subscribed is True

    @pytest.mark.asyncio
    async def test_on_host_offline_marks_unhealthy(self):
        """Test HOST_OFFLINE handler marks node unhealthy."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )

        event = MagicMock()
        event.payload = {"node_id": "bad_node", "reason": "timeout"}

        await selector._on_host_offline(event)

        assert "bad_node" in selector._unhealthy_nodes
        assert selector._unhealthy_reasons["bad_node"] == "timeout"

    @pytest.mark.asyncio
    async def test_on_host_online_clears_unhealthy(self):
        """Test HOST_ONLINE handler clears unhealthy status."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("recovered_node")
        selector._unhealthy_reasons["recovered_node"] = "was bad"

        event = MagicMock()
        event.payload = {"node_id": "recovered_node"}

        await selector._on_host_online(event)

        assert "recovered_node" not in selector._unhealthy_nodes

    @pytest.mark.asyncio
    async def test_on_node_recovered_clears_unhealthy(self):
        """Test NODE_RECOVERED handler clears unhealthy status."""
        selector = NodeSelector(
            get_peers=lambda: {},
            get_self_info=lambda: None,
        )
        selector._unhealthy_nodes.add("node1")

        event = MagicMock()
        event.payload = {"node_id": "node1"}

        await selector._on_node_recovered(event)

        assert "node1" not in selector._unhealthy_nodes
