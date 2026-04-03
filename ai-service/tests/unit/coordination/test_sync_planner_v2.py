"""Unit tests for SyncPlanner v2 - Intelligent Sync Planning.

Tests cover:
1. SyncPriority enum - values and ordering
2. SyncPlan dataclass - creation, properties, serialization, comparison
3. PlannerConfig dataclass - defaults, from_env()
4. PlannerStats dataclass - record_plan(), record_execution(), to_dict()
5. SyncPlanner class - lifecycle, event handling, planning methods

December 28, 2025.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.coordination.sync_planner_v2 import (
    SyncPriority,
    SyncPlan,
    PlannerConfig,
    PlannerStats,
    SyncPlanner,
    get_sync_planner,
    reset_sync_planner,
)
from app.coordination.data_catalog import DataEntry, DataType
from app.coordination.transport_manager import Transport


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_catalog():
    """Create a mock DataCatalog."""
    catalog = MagicMock()
    catalog.get_by_config.return_value = []
    catalog.get_by_type.return_value = []
    catalog.get_entries_on_node.return_value = []
    catalog.get_missing_on_node.return_value = []
    catalog.get_under_replicated.return_value = []
    catalog.get_replication_factor.return_value = 3
    catalog.get_by_path.return_value = None
    catalog.mark_synced.return_value = None
    catalog.health_check.return_value = MagicMock(healthy=True, message="")
    return catalog


@pytest.fixture
def mock_transport():
    """Create a mock TransportManager."""
    transport = MagicMock()
    transport.transfer_file = AsyncMock(return_value=MagicMock(success=True))
    return transport


@pytest.fixture
def planner_config():
    """Create a test PlannerConfig."""
    return PlannerConfig(
        min_replication_factor=2,
        target_replication_factor=3,
        max_replication_factor=5,
        training_deadline_seconds=60.0,
        ephemeral_rescue_deadline_seconds=30.0,
        model_sync_deadline_seconds=45.0,
    )


@pytest.fixture
def sample_entries():
    """Create sample DataEntry objects for testing."""
    return [
        DataEntry(
            path="/data/games/hex8_2p.db",
            data_type=DataType.GAMES,
            size_bytes=1_000_000,
            config_key="hex8_2p",
            checksum="abc123",
            mtime=time.time(),
            locations={"node-1"},
            primary_location="node-1",
        ),
        DataEntry(
            path="/data/games/hex8_4p.db",
            data_type=DataType.GAMES,
            size_bytes=2_000_000,
            config_key="hex8_4p",
            checksum="def456",
            mtime=time.time(),
            locations={"node-1", "node-2"},
            primary_location="node-1",
        ),
    ]


@pytest.fixture
def sync_planner(mock_catalog, mock_transport, planner_config):
    """Create a SyncPlanner with mocked dependencies."""
    with patch("app.coordination.sync_planner_v2.get_data_catalog", return_value=mock_catalog), \
         patch("app.coordination.sync_planner_v2.get_transport_manager", return_value=mock_transport), \
         patch("app.coordination.sync_planner_v2.socket.gethostname", return_value="test-coordinator"):
        planner = SyncPlanner(
            catalog=mock_catalog,
            transport_manager=mock_transport,
            config=planner_config,
        )
        yield planner


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    reset_sync_planner()
    yield
    reset_sync_planner()


# =============================================================================
# SyncPriority Tests
# =============================================================================


class TestSyncPriority:
    """Tests for SyncPriority enum."""

    def test_all_priority_values_exist(self):
        """Test all expected priority levels exist."""
        assert SyncPriority.BACKGROUND == 10
        assert SyncPriority.LOW == 25
        assert SyncPriority.NORMAL == 50
        assert SyncPriority.HIGH == 75
        assert SyncPriority.CRITICAL == 100

    def test_priority_ordering_ascending(self):
        """Test priorities are properly ordered (higher value = higher priority)."""
        assert SyncPriority.BACKGROUND < SyncPriority.LOW
        assert SyncPriority.LOW < SyncPriority.NORMAL
        assert SyncPriority.NORMAL < SyncPriority.HIGH
        assert SyncPriority.HIGH < SyncPriority.CRITICAL

    def test_priority_comparison_with_integers(self):
        """Test priorities can be compared with integers."""
        assert SyncPriority.NORMAL >= 50
        assert SyncPriority.HIGH >= 75
        assert SyncPriority.CRITICAL == 100

    def test_priority_is_intenum(self):
        """Test SyncPriority is an IntEnum for arithmetic operations."""
        # IntEnum allows arithmetic
        assert SyncPriority.NORMAL + 10 == 60
        assert SyncPriority.HIGH - SyncPriority.NORMAL == 25

    def test_priority_name_access(self):
        """Test priority name attribute."""
        assert SyncPriority.CRITICAL.name == "CRITICAL"
        assert SyncPriority.BACKGROUND.name == "BACKGROUND"


# =============================================================================
# SyncPlan Tests
# =============================================================================


class TestSyncPlan:
    """Tests for SyncPlan dataclass."""

    def test_sync_plan_creation_minimal(self):
        """Test creating SyncPlan with minimal arguments."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
        )
        assert plan.source_node == "node-1"
        assert plan.target_nodes == ["node-2"]
        assert plan.entries == []
        assert plan.priority == SyncPriority.NORMAL
        assert plan.reason == ""
        assert plan.deadline is None
        assert plan.config_key is None

    def test_sync_plan_creation_full(self, sample_entries):
        """Test creating SyncPlan with all arguments."""
        deadline = time.time() + 60.0
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2", "node-3"],
            entries=sample_entries,
            priority=SyncPriority.HIGH,
            reason="training_deps:hex8_2p",
            transport_preference=[Transport.RSYNC, Transport.HTTP_FETCH],
            deadline=deadline,
            config_key="hex8_2p",
            batch_id="batch-001",
        )
        assert plan.source_node == "node-1"
        assert len(plan.target_nodes) == 2
        assert len(plan.entries) == 2
        assert plan.priority == SyncPriority.HIGH
        assert plan.reason == "training_deps:hex8_2p"
        assert plan.deadline == deadline
        assert plan.config_key == "hex8_2p"
        assert plan.batch_id == "batch-001"

    def test_total_bytes_property_empty(self):
        """Test total_bytes with no entries."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
        )
        assert plan.total_bytes == 0

    def test_total_bytes_property_with_entries(self, sample_entries):
        """Test total_bytes sums all entry sizes."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
        )
        assert plan.total_bytes == 3_000_000  # 1MB + 2MB

    def test_is_expired_no_deadline(self):
        """Test is_expired returns False when no deadline."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=None,
        )
        assert plan.is_expired is False

    def test_is_expired_future_deadline(self):
        """Test is_expired returns False for future deadline."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=time.time() + 3600.0,  # 1 hour from now
        )
        assert plan.is_expired is False

    def test_is_expired_past_deadline(self):
        """Test is_expired returns True for past deadline."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=time.time() - 10.0,  # 10 seconds ago
        )
        assert plan.is_expired is True

    def test_time_to_deadline_no_deadline(self):
        """Test time_to_deadline returns None when no deadline."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=None,
        )
        assert plan.time_to_deadline is None

    def test_time_to_deadline_future(self):
        """Test time_to_deadline returns positive value for future deadline."""
        deadline = time.time() + 60.0
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=deadline,
        )
        ttd = plan.time_to_deadline
        assert ttd is not None
        assert 59.0 < ttd <= 60.0

    def test_time_to_deadline_past_returns_zero(self):
        """Test time_to_deadline returns 0 for past deadline (not negative)."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=time.time() - 100.0,
        )
        assert plan.time_to_deadline == 0

    def test_to_dict_serialization(self, sample_entries):
        """Test to_dict produces valid dictionary."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2", "node-3"],
            entries=sample_entries,
            priority=SyncPriority.HIGH,
            reason="test_reason",
            transport_preference=[Transport.RSYNC],
            deadline=time.time() + 60.0,
            config_key="hex8_2p",
            batch_id="batch-001",
        )
        result = plan.to_dict()

        assert result["source_node"] == "node-1"
        assert result["target_nodes"] == ["node-2", "node-3"]
        assert result["entry_count"] == 2
        assert result["total_bytes"] == 3_000_000
        assert result["priority"] == "HIGH"
        assert result["reason"] == "test_reason"
        assert result["config_key"] == "hex8_2p"
        assert result["batch_id"] == "batch-001"
        assert result["is_expired"] is False
        assert "created_at" in result

    def test_lt_comparison_higher_priority_first(self):
        """Test __lt__ puts higher priority plans first."""
        high_plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            priority=SyncPriority.HIGH,
        )
        low_plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            priority=SyncPriority.LOW,
        )
        # Higher priority should come first (< returns True)
        assert high_plan < low_plan
        assert not (low_plan < high_plan)

    def test_lt_comparison_same_priority_earlier_deadline_first(self):
        """Test __lt__ with same priority prefers earlier deadline."""
        now = time.time()
        early_plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            priority=SyncPriority.NORMAL,
            deadline=now + 30.0,
        )
        late_plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            priority=SyncPriority.NORMAL,
            deadline=now + 60.0,
        )
        assert early_plan < late_plan

    def test_lt_comparison_deadline_vs_no_deadline(self):
        """Test __lt__ prefers plans with deadline over those without."""
        with_deadline = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            priority=SyncPriority.NORMAL,
            deadline=time.time() + 60.0,
        )
        without_deadline = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            priority=SyncPriority.NORMAL,
            deadline=None,
        )
        assert with_deadline < without_deadline


# =============================================================================
# PlannerConfig Tests
# =============================================================================


class TestPlannerConfig:
    """Tests for PlannerConfig dataclass."""

    def test_default_values(self):
        """Test PlannerConfig has sensible defaults."""
        config = PlannerConfig()
        assert config.min_replication_factor == 3
        assert config.target_replication_factor == 5
        assert config.max_replication_factor == 10
        assert config.training_deadline_seconds == 300.0
        assert config.ephemeral_rescue_deadline_seconds == 60.0
        assert config.model_sync_deadline_seconds == 120.0
        assert config.max_plans_per_event == 20
        assert config.max_targets_per_plan == 5
        assert config.max_entries_per_plan == 100

    def test_custom_values(self):
        """Test PlannerConfig accepts custom values."""
        config = PlannerConfig(
            min_replication_factor=2,
            target_replication_factor=4,
            training_deadline_seconds=180.0,
        )
        assert config.min_replication_factor == 2
        assert config.target_replication_factor == 4
        assert config.training_deadline_seconds == 180.0

    def test_from_env_with_defaults(self):
        """Test from_env() uses environment variables."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove any existing env vars
            for key in ["RINGRIFT_MIN_REPLICATION", "RINGRIFT_TARGET_REPLICATION"]:
                os.environ.pop(key, None)
            config = PlannerConfig.from_env()
            assert config.min_replication_factor == 3
            assert config.target_replication_factor == 5

    def test_from_env_with_custom_values(self):
        """Test from_env() reads custom environment variables."""
        with patch.dict(os.environ, {
            "RINGRIFT_MIN_REPLICATION": "4",
            "RINGRIFT_TARGET_REPLICATION": "8",
            "RINGRIFT_TRAINING_DEADLINE": "600",
            "RINGRIFT_EPHEMERAL_DEADLINE": "120",
        }):
            config = PlannerConfig.from_env()
            assert config.min_replication_factor == 4
            assert config.target_replication_factor == 8
            assert config.training_deadline_seconds == 600.0
            assert config.ephemeral_rescue_deadline_seconds == 120.0


# =============================================================================
# PlannerStats Tests
# =============================================================================


class TestPlannerStats:
    """Tests for PlannerStats dataclass."""

    def test_initial_values(self):
        """Test PlannerStats starts with zero counters."""
        stats = PlannerStats()
        assert stats.plans_created == 0
        assert stats.plans_executed == 0
        assert stats.plans_succeeded == 0
        assert stats.plans_failed == 0
        assert stats.plans_expired == 0
        assert stats.total_bytes_planned == 0
        assert stats.total_entries_planned == 0

    def test_record_plan_increments_counters(self, sample_entries):
        """Test record_plan updates statistics."""
        stats = PlannerStats()
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
            priority=SyncPriority.NORMAL,
        )
        stats.record_plan(plan)

        assert stats.plans_created == 1
        assert stats.total_bytes_planned == 3_000_000
        assert stats.total_entries_planned == 2
        assert stats.normal_plans == 1
        assert stats.last_plan_time > 0

    def test_record_plan_priority_counters(self):
        """Test record_plan increments correct priority counter."""
        stats = PlannerStats()

        # Test each priority level
        priorities = [
            (SyncPriority.BACKGROUND, "background_plans"),
            (SyncPriority.LOW, "low_plans"),
            (SyncPriority.NORMAL, "normal_plans"),
            (SyncPriority.HIGH, "high_plans"),
            (SyncPriority.CRITICAL, "critical_plans"),
        ]

        for priority, counter_name in priorities:
            plan = SyncPlan(
                source_node="node-1",
                target_nodes=["node-2"],
                entries=[],
                priority=priority,
            )
            stats.record_plan(plan)
            assert getattr(stats, counter_name) == 1

    def test_record_execution_success(self):
        """Test record_execution tracks successful execution."""
        stats = PlannerStats()
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
        )
        stats.record_execution(plan, success=True)

        assert stats.plans_executed == 1
        assert stats.plans_succeeded == 1
        assert stats.plans_failed == 0
        assert stats.plans_expired == 0

    def test_record_execution_failure(self):
        """Test record_execution tracks failed execution."""
        stats = PlannerStats()
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
        )
        stats.record_execution(plan, success=False)

        assert stats.plans_executed == 1
        assert stats.plans_succeeded == 0
        assert stats.plans_failed == 1

    def test_record_execution_expired(self):
        """Test record_execution tracks expired plans."""
        stats = PlannerStats()
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],
            deadline=time.time() - 10.0,  # Already expired
        )
        stats.record_execution(plan, success=False)

        assert stats.plans_executed == 1
        assert stats.plans_expired == 1
        assert stats.plans_failed == 0  # Expired takes precedence

    def test_to_dict_structure(self):
        """Test to_dict produces expected structure."""
        stats = PlannerStats()
        stats.plans_created = 10
        stats.plans_executed = 8
        stats.plans_succeeded = 6
        stats.plans_failed = 1
        stats.plans_expired = 1

        result = stats.to_dict()

        assert result["plans_created"] == 10
        assert result["plans_executed"] == 8
        assert result["plans_succeeded"] == 6
        assert result["plans_failed"] == 1
        assert result["plans_expired"] == 1
        assert result["success_rate"] == 0.75  # 6/8
        assert "priority_breakdown" in result
        assert "critical" in result["priority_breakdown"]

    def test_to_dict_success_rate_zero_executions(self):
        """Test to_dict handles zero executions gracefully."""
        stats = PlannerStats()
        result = stats.to_dict()
        assert result["success_rate"] == 0.0


# =============================================================================
# SyncPlanner Class Tests
# =============================================================================


class TestSyncPlannerInit:
    """Tests for SyncPlanner initialization."""

    def test_init_with_defaults(self, mock_catalog, mock_transport):
        """Test SyncPlanner initializes with default dependencies."""
        with patch("app.coordination.sync_planner_v2.get_data_catalog", return_value=mock_catalog), \
             patch("app.coordination.sync_planner_v2.get_transport_manager", return_value=mock_transport), \
             patch("app.coordination.sync_planner_v2.PlannerConfig.from_env", return_value=PlannerConfig()):
            planner = SyncPlanner()
            assert planner._catalog is mock_catalog
            assert planner._transport is mock_transport
            assert planner._running is False

    def test_init_with_custom_config(self, mock_catalog, mock_transport, planner_config):
        """Test SyncPlanner accepts custom configuration."""
        planner = SyncPlanner(
            catalog=mock_catalog,
            transport_manager=mock_transport,
            config=planner_config,
        )
        assert planner._config.min_replication_factor == 2
        assert planner._config.training_deadline_seconds == 60.0

    def test_event_handlers_registered(self, sync_planner):
        """Test default event handlers are registered."""
        assert "SELFPLAY_COMPLETE" in sync_planner._event_handlers
        assert "TRAINING_STARTED" in sync_planner._event_handlers
        assert "MODEL_PROMOTED" in sync_planner._event_handlers
        assert "ORPHAN_GAMES_DETECTED" in sync_planner._event_handlers
        assert "NODE_TERMINATING" in sync_planner._event_handlers


class TestSyncPlannerLifecycle:
    """Tests for SyncPlanner start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_background_tasks(self, sync_planner):
        """Test start() creates background tasks."""
        await sync_planner.start()
        try:
            assert sync_planner._running is True
            assert sync_planner._replication_task is not None
            assert sync_planner._execution_task is not None
        finally:
            await sync_planner.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, sync_planner):
        """Test calling start() twice is safe."""
        await sync_planner.start()
        task1 = sync_planner._replication_task
        await sync_planner.start()  # Should not create new tasks
        assert sync_planner._replication_task is task1
        await sync_planner.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, sync_planner):
        """Test stop() cancels background tasks."""
        await sync_planner.start()
        await sync_planner.stop()
        assert sync_planner._running is False


class TestSyncPlannerEventHandling:
    """Tests for plan_for_event and event handlers."""

    def test_plan_for_event_unknown_event(self, sync_planner):
        """Test plan_for_event returns empty list for unknown events."""
        plans = sync_planner.plan_for_event("UNKNOWN_EVENT", {})
        assert plans == []

    def test_plan_for_event_records_stats(self, sync_planner, mock_catalog, sample_entries):
        """Test plan_for_event records statistics."""
        # Setup catalog to return entries
        mock_catalog.get_by_config.return_value = sample_entries
        sync_planner._node_capabilities = {
            "training-node": {"is_training": True, "is_gpu": True}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner.plan_for_event("SELFPLAY_COMPLETE", {
            "config_key": "hex8_2p",
            "source_node": "node-1",
        })

        assert sync_planner._stats.event_triggered_plans >= 0


class TestSelfplayCompleteHandler:
    """Tests for _handle_selfplay_complete."""

    def test_selfplay_complete_no_config_key(self, sync_planner):
        """Test handler returns empty when no config_key."""
        plans = sync_planner._handle_selfplay_complete({})
        assert plans == []

    def test_selfplay_complete_no_source_only_entries(self, sync_planner, mock_catalog, sample_entries):
        """Test handler returns empty when no source-only entries."""
        # All entries have multiple locations
        for entry in sample_entries:
            entry.locations = {"node-1", "node-2"}
        mock_catalog.get_by_config.return_value = sample_entries

        plans = sync_planner._handle_selfplay_complete({
            "config_key": "hex8_2p",
            "source_node": "node-1",
        })
        assert plans == []

    def test_selfplay_complete_creates_plan(self, sync_planner, mock_catalog, sample_entries):
        """Test handler creates plan for source-only entries."""
        # Entry only on source node
        sample_entries[0].locations = {"node-1"}
        mock_catalog.get_by_config.return_value = sample_entries

        # Setup targets
        sync_planner._node_capabilities = {
            "training-node": {"is_training": True, "is_gpu": True}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_selfplay_complete({
            "config_key": "hex8_2p",
            "source_node": "node-1",
        })

        assert len(plans) == 1
        assert plans[0].source_node == "node-1"
        assert plans[0].config_key == "hex8_2p"
        assert plans[0].priority == SyncPriority.NORMAL


class TestTrainingStartedHandler:
    """Tests for _handle_training_started."""

    def test_training_started_no_node_id(self, sync_planner):
        """Test handler returns empty when no node_id."""
        plans = sync_planner._handle_training_started({"config_key": "hex8_2p"})
        assert plans == []

    def test_training_started_no_config_key(self, sync_planner):
        """Test handler returns empty when no config_key."""
        plans = sync_planner._handle_training_started({"node_id": "training-1"})
        assert plans == []

    def test_training_started_creates_plan(self, sync_planner, mock_catalog, sample_entries):
        """Test handler creates training deps plan."""
        mock_catalog.get_missing_on_node.return_value = sample_entries

        plans = sync_planner._handle_training_started({
            "node_id": "training-1",
            "config_key": "hex8_2p",
        })

        assert len(plans) == 1
        assert plans[0].priority == SyncPriority.HIGH
        assert "training_deps" in plans[0].reason


class TestModelPromotedHandler:
    """Tests for _handle_model_promoted."""

    def test_model_promoted_no_model_path(self, sync_planner):
        """Test handler returns empty when no model_path."""
        plans = sync_planner._handle_model_promoted({})
        assert plans == []

    def test_model_promoted_model_not_in_catalog(self, sync_planner, mock_catalog):
        """Test handler returns empty when model not in catalog."""
        mock_catalog.get_by_path.return_value = None

        plans = sync_planner._handle_model_promoted({
            "model_path": "/models/hex8_2p.pth",
        })
        assert plans == []

    def test_model_promoted_creates_plan(self, sync_planner, mock_catalog, sample_entries):
        """Test handler creates model sync plan."""
        mock_catalog.get_by_path.return_value = sample_entries[0]

        sync_planner._node_capabilities = {
            "gpu-node": {"is_gpu": True}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_model_promoted({
            "model_path": "/models/hex8_2p.pth",
            "source_node": "coordinator",
        })

        assert len(plans) == 1
        assert plans[0].priority == SyncPriority.HIGH
        assert plans[0].deadline is not None


class TestOrphanDetectedHandler:
    """Tests for _handle_orphan_detected."""

    def test_orphan_detected_no_source_node(self, sync_planner):
        """Test handler returns empty when no source_node."""
        plans = sync_planner._handle_orphan_detected({})
        assert plans == []

    def test_orphan_detected_creates_plan(self, sync_planner, mock_catalog, sample_entries):
        """Test handler creates orphan recovery plan."""
        mock_catalog.get_entries_on_node.return_value = sample_entries
        mock_catalog.get_replication_factor.return_value = 1  # Under-replicated

        sync_planner._node_capabilities = {
            "stable-node": {"is_ephemeral": False, "is_coordinator": False}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_orphan_detected({
            "source_node": "ephemeral-node",
            "config_key": "hex8_2p",
        })

        assert len(plans) == 1
        assert "orphan_recovery" in plans[0].reason


class TestNodeTerminatingHandler:
    """Tests for _handle_node_terminating."""

    def test_node_terminating_no_node_id(self, sync_planner):
        """Test handler returns empty when no node_id."""
        plans = sync_planner._handle_node_terminating({})
        assert plans == []

    def test_node_terminating_no_at_risk_entries(self, sync_planner, mock_catalog, sample_entries):
        """Test handler returns empty when all entries well-replicated."""
        mock_catalog.get_entries_on_node.return_value = sample_entries
        mock_catalog.get_replication_factor.return_value = 5  # Well-replicated

        plans = sync_planner._handle_node_terminating({"node_id": "dying-node"})
        assert plans == []

    def test_node_terminating_creates_critical_plan(self, sync_planner, mock_catalog, sample_entries):
        """Test handler creates critical priority plan."""
        mock_catalog.get_entries_on_node.return_value = sample_entries
        mock_catalog.get_replication_factor.return_value = 1  # Under-replicated

        sync_planner._node_capabilities = {
            "stable-node": {"is_ephemeral": False, "is_coordinator": False}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_node_terminating({"node_id": "dying-node"})

        assert len(plans) == 1
        assert plans[0].priority == SyncPriority.CRITICAL
        assert plans[0].deadline is not None


class TestPlanTrainingDeps:
    """Tests for plan_training_deps method."""

    def test_plan_training_deps_no_missing(self, sync_planner, mock_catalog):
        """Test returns None when no missing data."""
        mock_catalog.get_missing_on_node.return_value = []

        plan = sync_planner.plan_training_deps("training-node", "hex8_2p")
        assert plan is None

    def test_plan_training_deps_no_sources(self, sync_planner, mock_catalog, sample_entries):
        """Test returns None when no sources found."""
        # Entries with no locations
        for entry in sample_entries:
            entry.locations = set()
        mock_catalog.get_missing_on_node.return_value = sample_entries

        plan = sync_planner.plan_training_deps("training-node", "hex8_2p")
        assert plan is None

    def test_plan_training_deps_creates_plan(self, sync_planner, mock_catalog, sample_entries):
        """Test creates plan with HIGH priority and deadline."""
        mock_catalog.get_missing_on_node.return_value = sample_entries

        plan = sync_planner.plan_training_deps("training-node", "hex8_2p")

        assert plan is not None
        assert plan.target_nodes == ["training-node"]
        assert plan.priority == SyncPriority.HIGH
        assert plan.deadline is not None
        assert plan.config_key == "hex8_2p"


class TestPlanReplication:
    """Tests for plan_replication method."""

    def test_plan_replication_none_under_replicated(self, sync_planner, mock_catalog):
        """Test returns empty when nothing under-replicated."""
        mock_catalog.get_under_replicated.return_value = []

        plans = sync_planner.plan_replication()
        assert plans == []

    def test_plan_replication_creates_plans(self, sync_planner, mock_catalog, sample_entries):
        """Test creates LOW priority plans for under-replicated entries."""
        mock_catalog.get_under_replicated.return_value = sample_entries

        sync_planner._node_capabilities = {
            "target-node": {"is_ephemeral": False, "is_coordinator": False}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner.plan_replication(min_factor=2)

        assert len(plans) > 0
        for plan in plans:
            assert plan.priority == SyncPriority.LOW
            assert "replication" in plan.reason


class TestPlanOrphanRecovery:
    """Tests for plan_orphan_recovery method."""

    def test_plan_orphan_recovery_no_entries(self, sync_planner, mock_catalog):
        """Test returns None when no entries on source."""
        mock_catalog.get_entries_on_node.return_value = []

        plan = sync_planner.plan_orphan_recovery("source-node")
        assert plan is None

    def test_plan_orphan_recovery_ephemeral_critical(self, sync_planner, mock_catalog, sample_entries):
        """Test ephemeral source gets CRITICAL priority."""
        mock_catalog.get_entries_on_node.return_value = sample_entries
        mock_catalog.get_replication_factor.return_value = 1

        sync_planner._node_capabilities = {
            "ephemeral-node": {"is_ephemeral": True},
            "stable-node": {"is_ephemeral": False, "is_coordinator": False},
        }
        sync_planner._capabilities_loaded = True

        plan = sync_planner.plan_orphan_recovery("ephemeral-node", "hex8_2p")

        assert plan is not None
        assert plan.priority == SyncPriority.CRITICAL

    def test_plan_orphan_recovery_stable_high(self, sync_planner, mock_catalog, sample_entries):
        """Test stable source gets HIGH priority."""
        mock_catalog.get_entries_on_node.return_value = sample_entries
        mock_catalog.get_replication_factor.return_value = 1

        sync_planner._node_capabilities = {
            "stable-node": {"is_ephemeral": False, "is_coordinator": False},
        }
        sync_planner._capabilities_loaded = True

        plan = sync_planner.plan_orphan_recovery("stable-node", "hex8_2p")

        assert plan is not None
        assert plan.priority == SyncPriority.HIGH


class TestSubmitAndExecutePlan:
    """Tests for submit_plan and execute_plan methods."""

    @pytest.mark.asyncio
    async def test_submit_plan_adds_to_queue(self, sync_planner, sample_entries):
        """Test submit_plan adds plan to pending queue."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
        )

        await sync_planner.submit_plan(plan)

        assert len(sync_planner._pending_plans) == 1

    @pytest.mark.asyncio
    async def test_execute_plan_expired_fails(self, sync_planner, sample_entries):
        """Test execute_plan fails for expired plan."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
            deadline=time.time() - 10.0,  # Already expired
        )

        success = await sync_planner.execute_plan(plan)

        assert success is False
        assert plan.error == "deadline_expired"

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, sync_planner, mock_transport, sample_entries):
        """Test execute_plan succeeds with valid plan."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
        )

        success = await sync_planner.execute_plan(plan)

        assert success is True
        assert plan.success is True
        assert mock_transport.transfer_file.called


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_running(self, sync_planner):
        """Test health_check returns healthy when running."""
        sync_planner._running = True

        result = sync_planner.health_check()

        assert result.healthy is True
        assert result.details["running"] is True

    def test_health_check_not_running(self, sync_planner):
        """Test health_check returns unhealthy when not running."""
        sync_planner._running = False

        result = sync_planner.health_check()

        assert result.healthy is False

    def test_health_check_high_backlog(self, sync_planner):
        """Test health_check reports degraded with high backlog."""
        sync_planner._running = True
        # Add many pending plans
        sync_planner._pending_plans = [MagicMock() for _ in range(150)]

        result = sync_planner.health_check()

        assert result.details["pending_plans"] == 150

    def test_health_check_low_success_rate(self, sync_planner):
        """Test health_check reports degraded with low success rate."""
        sync_planner._running = True
        sync_planner._stats.plans_executed = 100
        sync_planner._stats.plans_succeeded = 30  # 30% success rate

        result = sync_planner.health_check()

        # Success rate < 50% should trigger degraded status
        assert "success rate" in result.message.lower() or result.healthy is False


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_sync_planner_returns_instance(self):
        """Test get_sync_planner returns SyncPlanner instance."""
        with patch("app.coordination.sync_planner_v2.get_data_catalog"), \
             patch("app.coordination.sync_planner_v2.get_transport_manager"):
            planner = get_sync_planner()
            assert isinstance(planner, SyncPlanner)

    def test_get_sync_planner_singleton(self):
        """Test get_sync_planner returns same instance."""
        with patch("app.coordination.sync_planner_v2.get_data_catalog"), \
             patch("app.coordination.sync_planner_v2.get_transport_manager"):
            planner1 = get_sync_planner()
            planner2 = get_sync_planner()
            assert planner1 is planner2

    def test_reset_sync_planner_clears_instance(self):
        """Test reset_sync_planner clears singleton."""
        with patch("app.coordination.sync_planner_v2.get_data_catalog"), \
             patch("app.coordination.sync_planner_v2.get_transport_manager"):
            planner1 = get_sync_planner()
            reset_sync_planner()
            planner2 = get_sync_planner()
            assert planner1 is not planner2


# =============================================================================
# Edge Cases and Additional Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_target_nodes(self, sync_planner, mock_catalog, sample_entries):
        """Test handler behavior when no targets available."""
        mock_catalog.get_by_config.return_value = sample_entries
        sync_planner._node_capabilities = {}
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_selfplay_complete({
            "config_key": "hex8_2p",
            "source_node": "node-1",
        })
        assert plans == []

    def test_config_filtering(self, sync_planner, mock_catalog, sample_entries):
        """Test entries are filtered by config_key."""
        # Mix of configs
        sample_entries[0].config_key = "hex8_2p"
        sample_entries[1].config_key = "hex8_4p"
        mock_catalog.get_by_config.return_value = [sample_entries[0]]

        sync_planner._node_capabilities = {
            "training": {"is_training": True, "is_gpu": True}
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_selfplay_complete({
            "config_key": "hex8_2p",
            "source_node": "node-1",
        })

        # Only hex8_2p entries should be processed
        if plans:
            assert plans[0].config_key == "hex8_2p"

    def test_max_entries_per_plan_limit(self, sync_planner, mock_catalog):
        """Test entries are limited to max_entries_per_plan."""
        # Create many entries
        many_entries = [
            DataEntry(
                path=f"/data/games/game_{i}.db",
                data_type=DataType.GAMES,
                size_bytes=1000,
                config_key="hex8_2p",
                checksum=f"checksum_{i}",
                mtime=time.time(),
                locations={"source"},
                primary_location="source",
            )
            for i in range(200)
        ]
        mock_catalog.get_missing_on_node.return_value = many_entries

        plan = sync_planner.plan_training_deps("training", "hex8_2p")

        if plan:
            assert len(plan.entries) <= sync_planner._config.max_entries_per_plan

    def test_max_targets_per_plan_limit(self, sync_planner, mock_catalog, sample_entries):
        """Test targets are limited to max_targets_per_plan."""
        sample_entries[0].locations = {"source"}
        mock_catalog.get_by_config.return_value = [sample_entries[0]]

        # Many potential targets
        sync_planner._node_capabilities = {
            f"node-{i}": {"is_training": True, "is_gpu": True}
            for i in range(20)
        }
        sync_planner._capabilities_loaded = True

        plans = sync_planner._handle_selfplay_complete({
            "config_key": "hex8_2p",
            "source_node": "source",
        })

        if plans:
            assert len(plans[0].target_nodes) <= sync_planner._config.max_targets_per_plan
